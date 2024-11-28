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

import os
import datetime
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.extra_utils import random_id
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
from utils.neg_gauss_init_utils import  get_new_neg_points  # Code by lathika
import cv2 # Code by lathika - test
import numpy as np # Code by lathika - test

# Code by lathika - test
def check_for_nan(arr):
            return torch.isnan(arr.clone().detach()).any()
def test_each_iter_for_none(iteration, gaussians):
        with open("/home/lathika/Workspace/test/Dump/output_2.txt","a") as file:
            file.write(f"\n Iteration = {iteration} xyz = {check_for_nan(gaussians._xyz)}  op = {check_for_nan(gaussians._opacity[0])} \
                    f_dc = {check_for_nan(gaussians._features_dc)} f_r = {check_for_nan(gaussians._features_rest)} s = {check_for_nan(gaussians._scaling)} \
                        r = {check_for_nan(gaussians._rotation)} iP = {check_for_nan(gaussians._isPositive)} m_r = {check_for_nan(gaussians.max_radii2D)} xyz_g = {check_for_nan(gaussians.xyz_gradient_accum)}\
                        den = {check_for_nan(gaussians.denom)} xyz_grad = {gaussians._xyz.grad.data.norm(2).item()} op_grad = {gaussians._opacity.grad.data.norm(2).item()}\
                        f_dc_grad = {gaussians._features_dc.grad.data.norm(2).item()} f_r_grad = {gaussians._features_rest.grad.data.norm(2).item()} s_grad = {gaussians._scaling.grad.data.norm(2).item()}\
                        r_grad = {gaussians._rotation.grad.data.norm(2).item()} xyz_grad_accum = {check_for_nan(gaussians.xyz_gradient_accum)}  \n")



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, log_to_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    unwanted_gauss_total_acum = torch.zeros(size = (gaussians.get_xyz.shape[0],), dtype = torch.bool) # Code by lathika
    add_neg_gauss_flage = None # Code by lathika
    adding_neg_gauss_flag = None # Code by lathika - while adding neg gaussians

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

            # Code by lathika
            unwanted_gauss_acum = torch.zeros(size = (gaussians.get_xyz.shape[0],), dtype = torch.bool)
            if iteration >=1000:#1000
                if add_neg_gauss_flage== None:
                    add_neg_gauss_flage = True
                else:
                    add_neg_gauss_flage = False


        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, unwanted_gauss_filter, radii, means_2D, depths = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["unwanted_gauss_filter"], render_pkg["radii"], render_pkg["means_2D"], render_pkg["depths"]
                                                                        # Code by lathika - added " unwanted_gauss_filter, means_2D, depths"


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        # # Code by lathika - Test
        # if iteration==1000:
        #     print(f"max depth = {torch.max(depths)}")
        #     print(f"min depth = {torch.min(depths)}")
        #     print(f"num depth = {depths.shape[0]}")
        #     break

        # Code by lathika
        if iteration >=1000 and iteration%250==0: # Changed for testing   # After 1000 iterations, neg gaussians are added with 500 intervals
            add_neg_gauss_flage=None

        # Code by lathika
        if len(viewpoint_stack)==0:
            unwanted_gauss_total_acum = unwanted_gauss_acum

        # Code by lathika - test 
        if iteration%1000==0:
            print(iteration)
        it_1 = 7000
        it_2 = 30000
        it_3 = 40000
        
        if iteration == it_1 or iteration == it_2 or  iteration == it_3:
            image_cpu  = image.clone().detach().cpu().numpy()
            gt_cpu = gt_image.clone().cpu().numpy().transpose(1,2,0)
            #print(f"gt_image shape = {gt_cpu.shape}\n")
            #print(f"image shape = {image_cpu.shape}\n")
            image_cpu = image_cpu.transpose(1,2,0)
            #print(f"image shape = {image_cpu.shape}\n")
            image_cpu = cv2.cvtColor((image_cpu * 255).astype(np.uint8) , cv2.COLOR_BGR2RGB)
            gt_cpu = cv2.cvtColor((gt_cpu * 255).astype(np.uint8) , cv2.COLOR_BGR2RGB)

            cv2.imwrite(f"/home/lathika/Workspace/test/Dump/Dump_img/2_with_neg_test_1/output{iteration}.jpg", image_cpu)
            cv2.imwrite(f"/home/lathika/Workspace/test/Dump/Dump_img/2_with_neg_test_1/output_gt{iteration}.jpg", gt_cpu)

            with open("/home/lathika/Workspace/test/Dump/output.txt","a") as file:
                file.write(f"\n total_wanted_points = {torch.sum(~unwanted_gauss_filter)} , \n unwanted_points = {torch.sum(unwanted_gauss_filter)} \n")
                file.write(f"\n means_2D shape = {means_2D.shape}\n")
                file.write(f"depth shape = {depths.shape}\n")
                file.write(f"means_3D shape ={gaussians.get_xyz.shape}\n")
                file.write(f"features_dc shape = {gaussians._features_dc.shape}\n features_dc 10 = {gaussians._features_dc[:10]}\n")
                file.write(f"features_rest shape= {gaussians._features_rest.shape} \n features_rest 10 = {gaussians._features_rest[:10]}\n")
                file.write(f"opacity shape = {gaussians._opacity.shape} \n op_max ={torch.max(gaussians._opacity)} \n opacity 10 = {gaussians._opacity[:10]}\n")
                file.write(f"scaling shape = {gaussians.get_scaling.shape}\n scaling 10 = {gaussians.get_scaling[:10]}\n")
                file.write(f"rotation shape = {gaussians.get_rotation.shape}\n rotation 10 = {gaussians.get_rotation[:10]}\n")
                file.write(f"projection mat = {viewpoint_cam.full_proj_transform}\n")
                file.write(f"means_2D 10 = {means_2D[:10]}\n")
                file.write(f"depths 10 = {depths[:10]}\n")
                file.write(f"means_3D 10 = {gaussians.get_xyz[:10]}\n")
                file.write(f"image =\n {image_cpu}\n") # np.array2string(image_cpu, threshold=np.inf)
                file.write(f"gt_image =\n {gt_cpu}\n")


        # Code by lathika - test
        test_each_iter_for_none(iteration, gaussians)        

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            """
            # Code by lathika - For initializing the negative gaussians
            # Detect and get neg gauss points
            #add_neg_gauss_flage = False # Code by lathika to stop running the part below
            if add_neg_gauss_flage:
                if iteration % 30 == 0:     # This step is taken to reduce the number of images that will be processed for blob detection
                                            # Every image with an interval of 30, will be selected. (We can change this)
                    print(f"\n Going through neg gaus initialization at {iteration}")  # Test
                    print(f"\n Num of gauss before adding neg gauss = {gaussians._xyz.shape[0]}")
                    adding_neg_gauss_flag = True
                    neg_gaus_3d_means, blob_scales_tensor, blob_rotation = get_new_neg_points(gt_image, viewpoint_cam, means_2D, depths)

            """

            # Log and save
            training_report(log_to_wandb, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Code by lathika - test
            if iteration%10000==0:
                print(f"h= {int(viewpoint_cam.image_height)}, w = {int(viewpoint_cam.image_width)}, radii_len = {len(radii)}, \n total_wanted_points = {torch.sum(~unwanted_gauss_filter)} , \n unwanted_points = {torch.sum(unwanted_gauss_filter)} \n") 
                      # means_2D = {means_2D}, means_2DType = {type(means_2D)}, depths = {depths}, depths_type = {type(depths)} \n")
                print(f"max_x = {max(means_2D.cpu().numpy()[:,0])}, max_y= {max(means_2D.cpu().numpy()[:,1])}, \n min_x = {min(means_2D.cpu().numpy()[:,0])}, min_y= {min(means_2D.cpu().numpy()[:,1])} \n \
                      , min_depth = {min(depths.cpu().numpy())} ,max_depth = {max(depths.cpu().numpy())}, depth_len = {len(depths.cpu().numpy())} ")


            # Densification
            if iteration < opt.densify_until_iter:  
                
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # Code by lathika
                    """
                    if iteration > 1000 and iteration%200 ==0:
                        gaussians.prune_unwanted_gauss(unwanted_gauss_filter)    # Code by lathika
                        """
                    
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Code by lathika - For initializing the negative gaussians
            # Detect and get neg gauss points
            #add_neg_gauss_flage = False # Code by lathika to stop running the part below
            if add_neg_gauss_flage:
                if iteration % 30 == 0:     # This step is taken to reduce the number of images that will be processed for blob detection
                                            # Every image with an interval of 30, will be selected. (We can change this)
                    print(f"\n Going through neg gaus initialization at {iteration}")  # Test
                    print(f"\n Num of gauss before adding neg gauss = {gaussians._xyz.shape[0]}")
                    adding_neg_gauss_flag = True
                    neg_gaus_3d_means, blob_scales_tensor, blob_rotation = get_new_neg_points(gt_image, viewpoint_cam, means_2D, depths)
                    # Test
                    neg_gauss_mask = torch.rand(neg_gaus_3d_means.shape[0]) < 0.1 # Getting only 10%
                    neg_gaus_3d_means = neg_gaus_3d_means[neg_gauss_mask]
                    blob_scales_tensor = blob_scales_tensor[neg_gauss_mask]
                    blob_rotation = blob_rotation[neg_gauss_mask]


            # Code by lathika
            # Adding neg gaussians
            if adding_neg_gauss_flag:
                gaussians.add_neg_gauss(neg_gaus_3d_means, blob_scales_tensor, blob_rotation)
                print(f"\n Num of gauss after adding neg gauss = {gaussians._xyz.shape[0]}\n")
                adding_neg_gauss_flag = False

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(log_to_wandb, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if log_to_wandb:
        wandb.log({
            'train_loss_patches/l1_loss': Ll1.item(),
            'train_loss_patches/total_loss': loss.item(),
            'iter_time': elapsed,
            'scene/total_points': scene.gaussians.get_xyz.shape[0],
            # 'scene/small_points':(scene.gaussians.get_shape < 0.5).sum().item(),
            # 'scene/average_shape':scene.gaussians.get_shape.mean().item(),
            # # 'scene/large_shapes':scene.gaussians.get_shape[scene.gaussians.get_shape>=1.0].mean().item(),
            # 'scene/small_shapes':scene.gaussians.get_shape[scene.gaussians.get_shape<1.0].mean().item(),
            'scene/opacity_grads':scene.gaussians._opacity.grad.data.norm(2).item(),
            # 'scene/shape_grads':scene.gaussians._shape.grad.data.norm(2).item(),
        }, step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if log_to_wandb and (idx < 5):
                        wandb_key = f"renders/{config['name']}_view_{viewpoint.image_name}/render"
                        wandb.log({wandb_key: [wandb.Image(image, caption="Render at iteration {}".format(iteration))]}, step=iteration)
                        if iteration == testing_iterations[0]:
                            wandb_key = "renders/{}_view_{}/ground_truth".format(config['name'], viewpoint.image_name)
                            wandb.log({wandb_key: [wandb.Image(gt_image, caption="Ground truth at iteration {}".format(iteration))]}, step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if log_to_wandb:
                    wandb.log({
                        f"metrics/{config['name']}/loss_viewpoint - l1_loss": l1_test,
                        f"metrics/{config['name']}/loss_viewpoint - psnr": psnr_test,
                    }, step=iteration)

        if log_to_wandb:
            opacity_data = [[val] for val in scene.gaussians.get_opacity.cpu().squeeze().tolist()]
            wandb.log({
                "scene/opacity_histogram": wandb.plot.histogram(wandb.Table(data=opacity_data, columns=["opacity"]), "opacity", title="Opacity Histogram"),
                "scene/total_points": scene.gaussians.get_xyz.shape[0],
            }, step=iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--nowandb", action="store_false", dest='wandb')
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    exp_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # random_id()
    args.model_path = args.model_path + "_" + args.exp_set + "_" +  exp_id
    print("Optimizing " + args.model_path)
    safe_state(args.quiet, args.seed)
    setup = vars(args)
    setup["exp_id"] = exp_id
    if args.wandb:
        wandb_id = args.model_path.replace('outputs', '').lstrip('./').replace('/', '---')
        wandb.init(project="laplace", id=wandb_id, config = setup ,sync_tensorboard=False,settings=wandb.Settings(_service_wait=600))

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.wandb)

    # All done
    print("\nTraining complete.")
    if args.wandb:
        wandb.finish()