import argparse
from re import I
import time
from pathlib import Path
import yaml
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, \
    scale_coords, set_logging, increment_path, retrieve_image
from utils.torch_utils import select_device, time_synchronized
from utils.pose_utils import box_filter, get_3D_corners, pnp, get_camera_intrinsic, MeshPly

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

def detect(save_img=False):
    source, weights, view_img, imgsz, mesh_data, cam_intrinsics = opt.source, opt.weights, opt.view_img, opt.img_size, opt.mesh_data, opt.static_camera

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    torch.save(model.state_dict(), "state_dict_model.pt")
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if cam_intrinsics:
        with open(cam_intrinsics) as f:
            cam_intrinsics = yaml.load(f, Loader=yaml.FullLoader)

        dtx = np.array(cam_intrinsics["distortion"])
        mtx = np.array(cam_intrinsics["intrinsic"])

        fx = mtx[0,0]
        fy = mtx[1,1]
        u0 = mtx[0,2]
        v0 = mtx[1,2]


        internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)   

    save_img = True
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    
    mesh       = MeshPly(mesh_data)
    vertices   = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D  = get_3D_corners(vertices)

    # edges_corners = [[0, 1], [0, 3], [0, 7], [1, 2], [1, 6], [2, 3], [2, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7]]
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    colormap      = np.array(['r', 'g', 'b', 'c', 'm', 'y',  'k', 'w','xkcd:sky blue' ])

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    predictions = []
    t0 = time.time()
    count = 0
    for path, img, im0s, intrinsics, shapes in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Compute intrinsics
        if cam_intrinsics is None:
            fx, fy, det_height, u0, v0, im_native_width, im_native_height = intrinsics
            # fx, fy  = # calculate_focal_length(float(focal_len), int(im_native_width), int(im_native_height), float(det_width), float(det_height))
            internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        # Inference
        t1 = time_synchronized()
        pred, train_out = model(img, augment=False)
        # pred = model(img, augment=False)[0]

        # Using confidence threshold, eliminate low-confidence predictions
        pred = box_filter(pred, conf_thres=opt.conf_thres)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print(det)
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            (Path(str(save_dir / 'labels'))).mkdir(parents=True, exist_ok=True) 
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                det = det.cpu()
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :18], shapes[0], shapes[1])  # native-space pred
                prediction_confidence = det[i, 18]
                # box_pred = det.clone().cpu()
                box_predn = det[0, :18].clone().cpu()
                # Denormalize the corner predictions 
                corners2D_pr = np.array(np.reshape(box_predn, [9, 2]), dtype='float32')
                # Calculate rotation and tranlation in rodriquez format
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))
                pose_mat = cv2.hconcat((R_pr, t_pr))
                euler_angles = cv2.decomposeProjectionMatrix(pose_mat)[6]
                predictions.append([det, euler_angles, t_pr])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                if save_img:
                    # convert bgr to rgb
                
                    local_img       = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    figsize         = (shapes[0][1]/96, shapes[0][0]/96)
                    fig             = plt.figure(frameon=False, figsize=figsize)
                    ax              = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    # image = np.uint8(local_img) # .resize((im_native_width, im_native_height)))
                    ax.imshow(local_img, aspect='auto')
                    corn2D_pr= corners2D_pr[1:, :]

                    # Plot projection corners
                    for edge in edges_corners:
                        ax.plot(corn2D_pr[edge, 0], corn2D_pr[edge, 1], color='b', linewidth=0.5)
                    ax.scatter(corners2D_pr.T[0], corners2D_pr.T[1], c=colormap, s = 10)
                    
                    min_x, min_y = np.amin(corners2D_pr.T[0]), np.amin(corners2D_pr.T[1])

                    max_x, max_y = np.amax(corners2D_pr.T[0]), np.amax(corners2D_pr.T[1])

                    if False:
                        # Create a bounding box around the object
                        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, linewidth=1, edgecolor='r', facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)

                    ax.text(min_x, min_y-10, f"Conf: {prediction_confidence.cpu().numpy():.3f}, Rot: {euler_angles}", style='italic', bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
                    filename = os.path.basename(path).split('.')[0]+" "+ str(count)+"_predicted.png"
                    file_path = os.path.join(save_dir, filename)
                    fig.savefig(file_path, dpi = 96, bbox_inches='tight', pad_inches=0)
                    # fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
                    plt.close()

                    count += 1

                with open(txt_path + '.txt', 'a') as f:
                    f.write(str(det.numpy()) + '\n')

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                

    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l6_pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--static-camera', type=str, help='path to static camera intrinsics')
    parser.add_argument('--mesh-data', type=str, help='path to object specific mesh data')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
            detect()
