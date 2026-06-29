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
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    # max_split_size_mb:512 matches the training config.  128 is too small:
    # blocks >128 MB cannot be reused after a split, causing fragmentation.
    "garbage_collection_threshold:0.6,max_split_size_mb:512"
)
import json
import time
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx % 30:
            # continue
        if type(view) is list:
            view = view[0]
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        del rendering, gt
        if idx % 50 == 0:
            torch.cuda.empty_cache()

    if name == "test":
        t_list = []
        _views = list(views) if not isinstance(views, list) else views
        for view in tqdm(_views, desc="FPS measurement"):
            if type(view) is list:
                view = view[0]
            torch.cuda.synchronize()
            t_start = time.time()
            render(view, gaussians, pipeline, background)
            torch.cuda.synchronize()
            t_list.append(time.time() - t_start)
        import numpy as np
        t = np.array(t_list[5:])
        if len(t) > 0:
            xyz = gaussians.get_xyz
            fps = 1.0 / t.mean()
            print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')
            _results_path = os.path.join(model_path, "results.json")
            _existing = {}
            if os.path.exists(_results_path):
                with open(_results_path) as _f:
                    _existing = json.load(_f)
            _method_key = "ours_{}".format(iteration)
            if not isinstance(_existing.get(_method_key), dict):
                _existing[_method_key] = {}
            _existing[_method_key].update({"fps": float(fps), "num_gaussians": int(xyz.shape[0])})
            with open(_results_path, "w") as _f:
                json.dump(_existing, _f, indent=True)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.approx_l)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            if scene.use_loader:
                _render_workers = min(4, os.cpu_count() or 1)
                views = DataLoader(scene.getTrainCameras(), batch_size=1, shuffle=False, num_workers=_render_workers, collate_fn=list)
            else:
                views = scene.getTrainCameras()

            render_set(dataset.model_path, "train", scene.loaded_iter, views, gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)