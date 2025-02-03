"""
Filename: render_union
Incoporate EMOTE inference into GaussianAvatars inference pipeline
Merge orbit camera rendering feature
"""
import torch
from torch.utils.data import DataLoader
from scene import Scene
import os, json
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np
import librosa

from gaussian_renderer import render, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
from utils.viewer_utils import OrbitCamera
from mesh_renderer import NVDiffRenderer
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from arguments.stp import SplattingSettings

class EMOTEInference:
    """Handle EMOTE inference to get FLAME parameters."""
    
    def __init__(self, model_path, emote_ckpt_mode):
        from inferno_apps.TalkingHead.evaluation.TalkingHeadWrapper import TalkingHeadWrapper
        from inferno_apps.TalkingHead.evaluation.evaluation_functions import read_audio, process_audio, create_condition
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.talking_head = TalkingHeadWrapper(Path(model_path), ckpt_mode=emote_ckpt_mode, render_results=False).to(self.device)
        self.read_audio = read_audio
        self.process_audio = process_audio
        self.create_condition = create_condition
        
        self.TRAINING_IDS = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                            'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                            'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                            'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029']

    def run_inference(self, audio_path, subject_style='M003', emotion_idx=0, intensity_idx=2, neutral_mesh_path=''):
        # Set neutral mesh
        if neutral_mesh_path:
            print(f"Setting neutral mesh from {neutral_mesh_path}")
            from psbody.mesh import Mesh
            neutral_mesh = Mesh(filename=str(neutral_mesh_path))
            neutral_v = torch.from_numpy( neutral_mesh.v).to(dtype=torch.float32, device=self.talking_head.talking_head_model.device)
            self.talking_head.set_neutral_mesh(neutral_v)
        # Process audio
        wavdata, sampling_rate = self.read_audio(audio_path)
        sample = self.process_audio(wavdata, sampling_rate, video_fps=25)
        
        # Create base sample structure
        T = sample["raw_audio"].shape[0]
        reconstruction_type = self.talking_head.cfg.data.reconstruction_type[0] \
            if isinstance(self.talking_head.cfg.data.reconstruction_type, (list,)) \
            else self.talking_head.cfg.data.reconstruction_type
            
        sample["reconstruction"] = {reconstruction_type: {
            "gt_exp": torch.zeros((1, T, 50), dtype=torch.float32),
            "gt_shape": torch.zeros((1, 300), dtype=torch.float32),
            "gt_jaw": torch.zeros((1, T, 3), dtype=torch.float32),
            "gt_tex": torch.zeros((1, 50), dtype=torch.float32)
        }}
        
        # Set conditions
        identity_idx = self.TRAINING_IDS.index(subject_style)
        sample["samplerate"] = [sample["samplerate"]]
        sample["raw_audio"] = torch.tensor(sample["raw_audio"], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Add conditions
        sample = self.create_condition(self.talking_head, sample, 
                                     emotions=[emotion_idx],
                                     identities=[identity_idx],
                                     intensities=[intensity_idx])
        
        # Convert conditions and move to device
        for key in sample:
            if key.endswith('_condition') and isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key]).float().to(self.device)
        
        for key in sample["reconstruction"][reconstruction_type]:
            sample["reconstruction"][reconstruction_type][key] = sample["reconstruction"][reconstruction_type][key].to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.talking_head(sample)
        
        # Extract FLAME parameters
        flame_params = {
            "shape": output["gt_shape"][0].cpu().numpy(),
            "expression": output["predicted_exp"][0].cpu().numpy(),
            "jaw_pose": output["predicted_jaw"][0].cpu().numpy(),
            "global_pose": np.zeros_like(output["predicted_jaw"][0].cpu().numpy())
        }
        
        return flame_params

class RenderManager:
    """Handle rendering operations."""
    
    def __init__(self, render_type="fixed", width=960, height=540, radius=1, fovy=20):
        self.render_type = render_type
        self.mesh_renderer = NVDiffRenderer()
        if render_type == "orbit":
            self.orbit_cam = OrbitCamera(width, height, radius, fovy, convention="opencv")
    
    def sample_orbit_views(self, sequence_len: int, swing: float = 20.0):
        """Sample orbiting camera views."""
        orbit_views = []
        Dx_ = np.linspace(0, 2*3.14, 240)
        Dx = np.tile(Dx_, sequence_len)

        for i in tqdm(range(sequence_len), desc="Preparing views"):
            dx = swing * np.sin(Dx[i])
            dy = swing * np.cos(Dx[i])
            self.orbit_cam.orbit(dx, dy)
            view = self.orbit_cam.pop_view(reset=True)
            orbit_views.append(view)

        return orbit_views

    @staticmethod
    def write_outputs(path2data):
        """Write rendered outputs to disk."""
        for path, data in path2data.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if path.suffix in [".png", ".jpg"]:
                data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                Image.fromarray(data).save(path)
            elif path.suffix in [".obj", ".txt"]:
                with open(path, "w") as f:
                    f.write(data)
            elif path.suffix == ".npz":
                np.savez(path, **data)
            else:
                raise NotImplementedError(f"Unknown file type: {path.suffix}")

    def render_sequence(self, dataset, gaussians, pipeline, background, views, 
                   flame_params, render_path, render_mesh=False, audio_path=None, 
                   slow_eye_factor=1, save_views=False, splat_args=None):  # Add splat_args here
        """Render a sequence of frames."""
        views_data = {}
        worker_args = []
        max_threads = multiprocessing.cpu_count()
        
        new_exp = torch.tensor(flame_params['expression'])
        new_jaw = torch.tensor(flame_params['jaw_pose'])
        sequence_len = new_exp.shape[0]

        if self.render_type == "orbit":
            views = self.sample_orbit_views(sequence_len)
        else:
            view_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
            views = [list(view_loader)[0]] * sequence_len
        
        # for idx, view in enumerate(tqdm(views if isinstance(views, list) else [views], desc="Rendering progress")):
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # if gaussians.binding is not None:
                # gaussians.select_mesh_by_timestep(view.timestep)

            gaussians.update_mesh_by_param_expr_dict(
                new_exp[idx].reshape(1, -1),
                new_jaw[idx].reshape(1, -1),
                slow_eye_factor,
                fix_zero_pose=True,
                fix_zero_neck=True,
                idx=idx if hasattr(gaussians, 'binding') else None
            )

            rendering = render(view, gaussians, pipeline, background, splat_args=splat_args)["render"]
            
            path2data = {Path(render_path) / f'{idx:05d}.png': rendering}
            
            if render_mesh:
                out_dict = self.mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
                rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)
                rgb_mesh = rgba_mesh[:3, :, :]
                alpha_mesh = rgba_mesh[3:, :, :]
                mesh_opacity = 0.5
                rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity + rendering.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                path2data[Path(str(render_path) + '_mesh') / f'{idx:05d}.png'] = rendering_mesh

            worker_args.append([path2data])
            
            if self.render_type == "orbit":
                views_data[idx] = view.world_view_transform.tolist()

            if len(worker_args) >= max_threads:
                with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                    futures = [executor.submit(self.write_outputs, *args) for args in worker_args]
                    concurrent.futures.wait(futures)
                worker_args = []

        # Process any remaining frames
        if worker_args:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(self.write_outputs, *args) for args in worker_args]
                concurrent.futures.wait(futures)

        if save_views and views_data:
            with open(os.path.join(os.path.dirname(render_path), "views.json"), 'w') as f:
                json.dump(views_data, f, indent=4)

        # Create video with audio if audio path is provided
        if audio_path:
            frame_rate = 25  # EMOTE default frame rate
            video_name = render_path.parent.stem
            output_video = os.path.join(os.path.dirname(render_path), f"{video_name}.mp4")
            os.system(f"ffmpeg -y -framerate {frame_rate} -f image2 -pattern_type glob -i '{render_path}/*.png' "
                     f"-i {audio_path} -b:v 5M -pix_fmt yuv420p {output_video}")

def main():
    parser = ArgumentParser(description="Gaussian Avatar Rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # Add render-specific arguments
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--render_type", choices=["fixed", "orbit"], default="fixed")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--emote_model", type=str, default="inferno_apps/TalkingHead/pretrained/EMOTE_v2")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--subject_style", type=str, default='M003')
    parser.add_argument("--emotion_idx", type=int, default=0)
    parser.add_argument("--intensity_idx", type=int, default=0)
    parser.add_argument("--expname", type=str, default='renders')
    parser.add_argument("--emote_ckpt_mode", type=str, default='latest')
    parser.add_argument("--neutral_mesh_path", type=str, default='')
    
    args = get_combined_args(parser)
    splat_args = SplattingSettings(parser).get_settings(args)

    if args.render_type != 'orbit':
        assert args.select_camera_id != -1, "Not orbit -> camera ID must be provided"
    
    # Initialize renderers and EMOTE
    render_manager = RenderManager(render_type=args.render_type)
    emote = EMOTEInference(args.emote_model, args.emote_ckpt_mode)
    print(f'Initialized EMOTE with ckpt mode <{args.emote_ckpt_mode}>')
    
    with torch.no_grad():
        # Initialize Gaussian model
        gaussians = FlameGaussianModel(args.sh_degree) if args.bind_to_mesh else GaussianModel(args.sh_degree)
        
        # Set up scene and background
        scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
        background = torch.tensor([1,1,1] if args.white_background else [0,0,0], dtype=torch.float32, device="cuda")
        
        # Get FLAME parameters from EMOTE
        flame_params = emote.run_inference(
            args.audio, 
            args.subject_style,
            args.emotion_idx,
            args.intensity_idx,
            args.neutral_mesh_path
        )
        
        # Set up output paths
        output_dir = Path(args.model_path) / args.expname
        render_path = output_dir / "renders"
        render_path.mkdir(parents=True, exist_ok=True)
        
        # Render sequence
        render_manager.render_sequence(
            model.extract(args),
            gaussians,
            pipeline.extract(args),
            background,
            scene.getTrainCameras(),
            flame_params,
            render_path,
            args.render_mesh,
            args.audio,
            save_views=(args.render_type == "orbit"),
            splat_args=splat_args
        )

if __name__ == "__main__":
    main()
    """
    python render_union.py \ 
    --model_path output/nguyen_v24_sh1/ \ 
    --audio audios/japanese.wav \ 
    --render_type orbit \ 
    --expname test_fix_render \ 
    --iteration 360000 \ 
    --emote_ckpt_mode finetuned
    """