#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer

### for stp
from diff_gaussian_rasterization import ExtendedSettings
from arguments.stp import SplattingSettings
###


mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh, flame_file, audio_path, mode, expname, splat_args):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = os.path.join(iter_path, expname)
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = os.path.join(iter_path, expname)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []

    ##########################
    ## METHOD 1 - DECA ##
    # new_flame_param = np.load('./data/chinese_groundtruth/{}.npz'.format(idx))
    # gaussians.update_mesh_by_param_expr_dict(torch.tensor(new_flame_param['expr']), torch.tensor(new_flame_param['jaw_pose']))
    if mode == 'emote':

        ## METHOD 2 - EMOTE ## 
        new_flame_param = np.load(
            file=flame_file,
            allow_pickle=True)
        new_exp = torch.tensor(new_flame_param['expression'])
        new_jaw = torch.tensor(new_flame_param['jaw_pose'])
        # new_neck = torch.tensor(new_flame_param['global_pose'])
        frame_rate = 25
        slow_eye_factor = 1

    elif mode == 'voca':
        ## METHOD 3 - VOCA ## 
        new_flame_param = np.load(
            file=flame_file,
            allow_pickle=True).item()
        new_exp = torch.tensor(new_flame_param['expression'], dtype=torch.float32)
        new_jaw = torch.tensor(new_flame_param['pose'][:, 6:9], dtype=torch.float32) 
        # scale jaw pose (only scale R_x-axis when it's positive)
        new_jaw[:, 0][new_jaw[:,0] > 0] *= 2
        # new_neck = torch.tensor(new_flame_param['global_pose'])
        frame_rate = 60
        slow_eye_factor = 2

    size = new_exp.shape[0] 
    fix_init_view = True

    # Fix view real
    view = list(views_loader)[0]

    for idx in (tqdm(range(size), desc="Rendering progress")):
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)

        # if idx == 0:
        #     same_view=view
        # if idx == size:
        #     break

        gaussians.update_mesh_by_param_expr_dict(
            new_exp[idx].reshape(1, -1),
            new_jaw[idx].reshape(1, -1),
            slow_eye_factor,
            fix_zero_pose=True,
            fix_zero_neck=True,
            idx=idx)
        
        ##########################
        if fix_init_view:
            rendering = render(views[0], gaussians, pipeline, background, splat_args=splat_args)["render"]
        else:
            rendering = render(view, gaussians, pipeline, background, splat_args=splat_args)["render"]
        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.system(f"ffmpeg -y -framerate {frame_rate} -f image2 -pattern_type glob -i '{render_path}/*.png' -i {audio_path} -b:v 5M -pix_fmt yuv420p {iter_path}/{expname}.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate {frame_rate} -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -b:v 5M -pix_fmt yuv420p {iter_path}/{expname}_mesh.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool, is_debugging:bool, flame_file:str, audio_path:str, mode:str, expname: str, splat_args: ExtendedSettings):

    assert mode in ['emote', 'voca']

    with torch.no_grad():
        if dataset.bind_to_mesh:
            # gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset)
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, flame_file, mode, expname, splat_args=splat_args)
        else:
            """
            Modify to get custom views here
            """
            if is_debugging: # Later change to "if custom_view:"
                test_cameras = scene.getTestCameras()
                train_cameras = scene.getTrainCameras()
                val_cameras = scene.getValCameras()
            else:
                if not skip_train:
                    render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, flame_file, audio_path, mode, expname, splat_args=splat_args)
                
                if not skip_val:
                    render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh, flame_file, audio_path, mode, expname, splat_args=splat_args)

                if not skip_test:
                    render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh, flame_file, audio_path, mode, expname, splat_args=splat_args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--debug_flag", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--flame", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--expname", type=str, default='renders')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    ## for stp
    ss = SplattingSettings(parser)
    splat_args = ss.get_settings(args)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # models = model.extract(args)
    # print("source", models.source_path) # ./data/306/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine
    # print("target", models.target_path)
    # print("model", models.model_path)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh, args.debug_flag, args.flame, args.audio, args.mode, args.expname, splat_args)