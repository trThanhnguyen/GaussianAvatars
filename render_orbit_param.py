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
from scene import Scene
import os, json
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
from utils.viewer_utils import OrbitCamera


### for stp
from diff_gaussian_rasterization import ExtendedSettings
from arguments.stp import SplattingSettings
###


# Camera configs #
W = 960
H = 540
r = 1
fovy = 20

# W = 550
# H = 802
# r = 1
# fovy = 0.8

mesh_renderer = NVDiffRenderer()

def sample_orbit_views(orbit_cam: OrbitCamera, sq_len: int, swing: float):
    """
    Sample a list of orbiting views, look_at point = [0, 0, 0]
    Edit your camera patterns here
    """
    orbit_views = []

    # Example pattern:
    # Circle camera around throughout 240 time steps, magnitude defined by swing variable
    Dx_ = np.linspace(0, 2*3.14, 240)
    Dx = np.tile(Dx_, sq_len)

    for i in tqdm(range(sq_len), desc= "Preparing views"):
        dx = swing * np.sin(Dx[i])
        dy = swing * np.cos(Dx[i])
        orbit_cam.orbit(dx, dy)
        view = orbit_cam.pop_view(reset=True)
        # Reset to original camera pose after each iter
        # If not, changes are additive
        # Depends on your choice of pattern

        orbit_views.append(view)
        # swing += 1

    return orbit_views

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

def render_set(dataset : ModelParams, gaussians, pipeline, background, render_mesh, flame_file, audio_path, mode, runname, splat_args):
    
    save_dir = os.path.join(Path(dataset.model_path) , runname)
    print("Save Dir: ", save_dir)
    render_path = os.path.join(save_dir, "renders")
    if render_mesh:
        render_mesh_path = os.path.join(save_dir + 'mesh', runname)

    makedirs(render_path, exist_ok=True)

    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []


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
        new_jaw = torch.tensor(new_flame_param['pose'][:, 6:9], dtype=torch.float32) # scale jaw pose
        new_jaw[:, 0][new_jaw[:,0] > 0] *= 2
        # new_neck = torch.tensor(new_flame_param['global_pose'])
        frame_rate = 60
        slow_eye_factor = 2

    orbit_cam = OrbitCamera(W, H, r, fovy, convention="opencv")
    sequence_len = new_exp.shape[0]

    views = sample_orbit_views(orbit_cam, sequence_len, swing=170.0)
    ###### should be a function
    ################### 

    view_dic = {}    

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        gaussians.update_mesh_by_param_expr_dict(
            new_exp[idx].reshape(1, -1),
            new_jaw[idx].reshape(1, -1),
            slow_eye_factor,
            fix_zero_pose=True, # True False
            fix_zero_neck=True)
        
        rendering = render(view, gaussians, pipeline, background, splat_args=splat_args)["render"]

        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering

        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []

        view_dic[idx] = view.world_view_transform.tolist()
    
    with open(os.path.join(save_dir, "views.json"), 'w') as json_file:
        json.dump(view_dic, json_file, indent=4)

    try:
        os.system(f"ffmpeg -y -framerate {frame_rate} -f image2 -pattern_type glob -i '{render_path}/*.png' -i {audio_path} -pix_fmt yuv420p -b:v 5M {save_dir}/render_{runname}.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_mesh: bool, is_debugging:bool, flame_file:str, audio: str, mode:str, runname: str, splat_args: ExtendedSettings):

    assert mode in ['emote', 'voca']

    with torch.no_grad():
        if dataset.bind_to_mesh:
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # no need to load cameras from dataset
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset, gaussians, pipeline, background, render_mesh, flame_file, audio, mode, runname, splat_args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--debug_flag", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--flame", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--runname", type=str, default='renders')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    ## for stp
    ss = SplattingSettings(parser)
    splat_args = ss.get_settings(args)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.render_mesh, args.debug_flag, args.flame, args.audio, args.mode, args.runname, splat_args)