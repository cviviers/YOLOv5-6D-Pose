import argparse
import json
import os
from pathlib import Path
from threading import Thread
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm
from  torch.cuda.amp import autocast
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, scale_coords, set_logging, increment_path, colorstr, retrieve_image
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from utils.pose_utils import box_filter, get_3D_corners, pnp, epnp, calcAngularDistance, compute_projection, compute_transformation, get_camera_intrinsic, fix_corner_order, calc_pts_diameter, MeshPly
from utils.loss import PoseLoss
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

from utils.compute_overlap import wrapper_c_min_distances # for computing ADD-S metric

def test(data, weights=None, batch_size=1,
         imgsz=640,
         conf_thres=0.01,
         num_keypoints = 9,
         save_json=False,
         single_cls=True,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         nc = 1,
         log_imgs=0,  # number of logged images
         compute_loss=False,
         symetric = False,
         test_plotting = False):

    # Initialize/load model and set device
    training = model is not None


    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    
    testing_error_trans  = 0.0
    testing_error_angle  = 0.0
    testing_error_pixel  = 0.0
    testing_samples      = 0.0
    errs_2d              = []
    errs_3d              = []
    errs_trans           = []
    errs_angle           = []
    errs_corner2D        = []



    # Variable to save
    testing_errors_trans    = []
    testing_errors_angle    = []
    testing_errors_pixel    = []
    testing_accuracies      = []

    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    colormap      = np.array(['r', 'g', 'b', 'c', 'm', 'y',  'k', 'w','xkcd:sky blue' ])
    
    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, gs, opt, rect=True, augment=False,
                                       prefix=colorstr('test: ' if opt.task == 'test' else 'val: '))[0]

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    t1, t2, t3, t4, t5, t6 = [], [], [],[], [], []
    if compute_loss:
        pose_loss = PoseLoss(model, num_keypoints, pretrain_num_epochs=0)
    loss_items = torch.zeros(3, device=device, requires_grad=False)
    loss = torch.zeros(1, device=device, requires_grad=False)
    # Get the intrinsic camerea matrix, mesh, vertices and corners of the model
    # mesh_list = []
    # for mesh_id in range(8):
    #     mesh_list.append(MeshPly(data[f'mesh{mesh_id}']))

    mesh       = MeshPly(data[f'mesh'])
    vertices   = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D  = get_3D_corners(vertices)

    try:
        diam  = float(data['diam'])
    except:
        diam  = calc_pts_diameter(np.array(mesh.vertices))

    wandb_images = []
    count = 0

    for batch_i, (img, targets, intrinsics, paths, shapes) in enumerate(tqdm(dataloader)):
        t = time_synchronized()
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with torch.no_grad():
            # Run model
            with autocast():
                t1.append(time_synchronized() - t)

                out, train_out = model(img, augment=augment)  # inference and training outputs
                t2.append(time_synchronized() - t)

            # Compute loss
            if compute_loss:
                # _, loss_items = pose_loss([x.float() for x in train_out], targets, imgs_size=list(img.shape[2:]))
                batch_loss, batch_loss_items = pose_loss([x.float() for x in train_out], targets)
                loss_items += batch_loss_items  
                loss += batch_loss

            t3.append(time_synchronized() - t)

            # Using confidence threshold, eliminate low-confidence predictions
            out = box_filter(out, conf_thres=conf_thres)
            t4.append(time_synchronized() - t)

            # Statistics per image
            for si, pred in enumerate(out):
  
                path, shape = Path(paths[si]), shapes[si][0]
                im_native_width, im_native_height = shape[1], shape[0]
                # Predictions
                if len(pred) == 0:
                    continue
                if single_cls:
                    pred[:, 19] = 0
                predn = pred.clone().cpu()
                scale_coords(img[si].shape[1:], predn[:, :18], shape, shapes[si][1])  # native-space pred
                labels = targets[targets[:, 0] == si, 1:].cpu()
                tbox = labels[: ,1:19]
                tbox[:, ::2] = tbox[:, ::2]*width
                tbox[:, 1::2] = tbox[:, 1::2]*height
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels

                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target classes
                
                seen += 1
                # Iterate through each prediction and ground-truth object

                for k in range(nl):

                    box_gt = tbox[k]
                    full_pr = predn[torch.where(predn[:, 19] == k), :]
                    if len(full_pr) == 0 or not full_pr.shape[0] or full_pr.nelement()==0:
                        continue
                    
                    box_pr = full_pr[0,:18]
                    prediction_confidence = full_pr[0,18]
                    # Denormalize the corner predictions 
                    corners2D_gt = np.array(np.reshape(box_gt[:num_keypoints*2], [num_keypoints, 2]), dtype='float32')
                    corners2D_pr = np.array(np.reshape(box_pr[:num_keypoints*2], [num_keypoints, 2]), dtype='float32')

                    # Compute corner prediction error
                    corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                    corner_dist = np.mean(corner_norm)
                    errs_corner2D.append(corner_dist)

                    u0, v0, fx, fy = intrinsics[k][4], intrinsics[k][5], intrinsics[k][0], intrinsics[k][1]
                    internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

                    # Compute [R|t] by pnp
                    R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_gt, np.array(internal_calibration, dtype='float32'))
                    t_temp = time_synchronized()
                    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))
                    t6.append(time_synchronized() - t_temp)

                    # Compute errors
                    # Compute translation error
                    trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                    errs_trans.append(trans_dist)

                    # Compute angle error
                    angle_dist   = calcAngularDistance(R_gt, R_pr)
                    errs_angle.append(angle_dist)

                    # Compute pixel error
                    Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                    Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                    proj_2d_gt   = compute_projection(vertices, Rt_gt, internal_calibration) 
                    proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration) 
                    norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                    pixel_dist   = np.mean(norm)
                    errs_2d.append(pixel_dist)

                    # Compute 3D distances
                    transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
                    transform_3d_pred = compute_transformation(vertices, Rt_pr)  
                    if symetric:
                        norm3d         = wrapper_c_min_distances(transform_3d_gt, transform_3d_pred)
                    else:
                        norm3d        = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                    vertex_dist       = np.mean(norm3d)    
                    errs_3d.append(vertex_dist)  

                    # Sum errors
                    testing_error_trans  += trans_dist
                    testing_error_angle  += angle_dist
                    testing_error_pixel  += pixel_dist
                    testing_samples      += 1

                    # test_plotting = False
                    # W&B logging
                    if test_plotting or (plots and len(wandb_images)) < log_imgs:

                        local_img           = img[si, : , : , :].cpu().numpy().transpose(1, 2, 0)
                        local_img           = retrieve_image(local_img, img[si].shape[1:], (shape[0], shape[1]), shapes[si][1]) #  im_native_width, im_native_height 
                        figsize=(im_native_width/96, im_native_height/96)
                        fig = plt.figure(frameon=False, figsize=figsize)

                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        
                        image = np.uint8(local_img*255) # .resize((im_native_width, im_native_height)))
                        ax.imshow(image, cmap='gray', aspect='auto')
                       
                        corn2D_pr= corners2D_pr[1:, :]
                        corn2D_gt= corners2D_gt[1:, :]

                        # Projections
                        for edge in edges_corners:
                            ax.plot(corn2D_gt[edge, 0], corn2D_gt[edge, 1], color='g', linewidth=0.5) #  if test_plotting else None
                            ax.plot(corn2D_pr[edge, 0], corn2D_pr[edge, 1], color='b', linewidth=0.5)
                        ax.scatter(corners2D_gt.T[0], corners2D_gt.T[1], c=colormap, s = 10)  # if not test_plotting else None
                        ax.scatter(corners2D_pr.T[0], corners2D_pr.T[1], c=colormap, s = 10)

                        # draw on image
                        # Create a Rectangle patch
                        min_x = np.amin(corners2D_pr.T[0])
                        min_y = np.amin(corners2D_pr.T[1])

                        # vx_threshold = diam * 0.1
                        # facecolor = 'green' if vertex_dist <=vx_threshold else 'red'
                        # ax.text(min_x, min_y-30, f"conf: {prediction_confidence:.3f}", style='italic', bbox={'facecolor': facecolor, 'alpha': 0.5, 'pad': 2})
                        # ax.text(min_x, min_y-10, f"2d err: {pixel_dist:.3f}, vertex_dist: {vertex_dist:.3f}", style='italic', bbox={'facecolor': facecolor, 'alpha': 0.5, 'pad': 2})
                        

                        filename = f'foo_{count}_{datetime.now().strftime("%H_%M_%S")}.png'
                        file_path = os.path.join(save_dir, filename)
                        fig.savefig(file_path, dpi = 96, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        wandb_images.append(wandb.Image(file_path)) if not test_plotting else None

                        count+=1
            t5.append(time_synchronized() - t)

    # Compute 2D projection, 6D pose and 5cm5degree scores
    px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works 
    vx_threshold = diam * 0.1
    eps          = 1e-5
    acc_value    = len(np.where(np.array(errs_2d) <= px_threshold)[0]) 
    acc          = acc_value * 100. / (len(errs_2d)+eps) 
    acc3d_value  = len(np.where(np.array(errs_3d) <= vx_threshold)[0]) 
    acc3d        = acc3d_value * 100. / (len(errs_3d)+eps) 
    acc5cm5deg_value   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) 
    acc5cm5deg   = acc5cm5deg_value* 100. / (len(errs_trans)+eps)
    corner_acc   = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
    mean_err_2d  = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)
    
    t1 = np.array(t1)
    t2 = np.array(t2)
    t3 = np.array(t3)
    t4 = np.array(t4)
    t5 = np.array(t5)
    t6 = np.array(t6)

    num_itr = 5 # first couple of passes are slow
    if True:
        print('-----------------------------------')
        print('  tensor to cuda : %f' % (np.mean(t1[-num_itr:])))
        print('         predict : %f' % (np.mean((t2 - t1)[-num_itr:])))
        print('    compute loss : %f' % (np.mean((t3 - t2)[-num_itr:])))
        print('get_region_boxes : %f' % (np.mean((t4 - t3)[-num_itr:])))
        print('            eval : %f' % (np.mean((t5 - t4)[-num_itr:])))
        print('             pnp : %f' % (np.mean(t6[-num_itr:])))
        print('           total : %f' % (np.mean(t4[-num_itr:])))
        print('-----------------------------------')

    # Print test statistics
    print("   Mean corner error is %f" % (mean_corner_err_2d))
    print('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    print('   Acc using {} vx 3D Transformation = {:.2f}%'.format(vx_threshold, acc3d))
    print('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    print('   Translation error: %f, angle error: %f' % (testing_error_trans/(nts+eps), testing_error_angle/(nts+eps)) )

    # Register losses and errors for saving later on
    testing_errors_trans.append(testing_error_trans/(nts+eps))
    testing_errors_angle.append(testing_error_angle/(nts+eps))
    testing_errors_pixel.append(testing_error_pixel/(nts+eps))
    testing_accuracies.append(acc)
    # Return results
    model.float()  # for training

    # Plots
    if plots:
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})

    return (mean_corner_err_2d, acc, acc3d, acc5cm5deg, *(loss_items.cpu().detach()/ len(dataloader)).tolist(), loss.cpu().numpy().item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--num-keypoints', type=int, default=9, help='number keypoints')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--symetric', action='store_true', help='symetric object')
    parser.add_argument('--test-plotting', action='store_true', help='plot all predictions on test images')

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements()

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.num_keypoints,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             symetric=opt.symetric,
             test_plotting=opt.test_plotting
             )