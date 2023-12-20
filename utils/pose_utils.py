import time
import os
import math
import torch
import numpy as np
import cv2
from scipy import spatial
from matplotlib.path import Path as mat_Path

def get_all_files(directory):
    files = []

    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
        else:
            files.extend(get_all_files(os.path.join(directory, f)))
    return files

def calcAngularDistance(gt_rot, pr_rot):

    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff) 
    return np.rad2deg(np.arccos((trace-1.0)/2.0))

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d


def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def adi(pts_est, pts_gt):
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e

def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)

def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32') 

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D, np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)), cameraMatrix, distCoeffs)                            

    R, _ = cv2.Rodrigues(R_exp)
    return R, t

def epnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32') 

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'


    _, R_exp, t = cv2.solvePnP(points_3D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs,
                              flags = cv2.SOLVEPNP_EPNP)    

    R, _ = cv2.Rodrigues(R_exp)
    return R, t

def PnPRansac(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32') 

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t, _ = cv2.solvePnPRansac(points_3D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs,
                              flags = cv2.SOLVEPNP_EPNP)     

    R, _ = cv2.Rodrigues(R_exp)
    return R, t

def fix_corner_order(corners2D_gt):
    corners2D_gt_corrected = np.zeros((9, 2), dtype='float32')
    corners2D_gt_corrected[0, :] = corners2D_gt[0, :]
    corners2D_gt_corrected[1, :] = corners2D_gt[1, :]
    corners2D_gt_corrected[2, :] = corners2D_gt[3, :]
    corners2D_gt_corrected[3, :] = corners2D_gt[5, :]
    corners2D_gt_corrected[4, :] = corners2D_gt[7, :]
    corners2D_gt_corrected[5, :] = corners2D_gt[2, :]
    corners2D_gt_corrected[6, :] = corners2D_gt[4, :]
    corners2D_gt_corrected[7, :] = corners2D_gt[6, :]
    corners2D_gt_corrected[8, :] = corners2D_gt[8, :]
    return corners2D_gt_corrected


def corner_confidence(gt_corners, pr_corners, im_grid_width, im_grid_height, th=0.25, sharpness=2, device=None):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values 
    '''
    
    th = th*(np.sqrt(im_grid_width**2+im_grid_height**2) )
    dist = gt_corners - pr_corners
    dist = torch.reshape(dist, (-1, 9, 2))

    eps = 1e-5
    dist  = torch.sqrt(torch.sum((dist)**2, dim=2))
    mask  = (dist < th).type(torch.FloatTensor).to(device)
    conf  = (torch.exp(sharpness * (1.0 - dist/th)) - 1).to(device)
    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1 + eps
    conf0 = conf0.to(device)
    conf  = conf / conf0.repeat(1, 9)
    conf  = mask * conf 
    return torch.mean(conf, dim=1)


def box_filter(prediction, conf_thres=0.01, classes=None, multi_label=False, max_det = 1):
    """Performs box filtering on inference results

    Returns:
         detections with shape: nx6 (x0, y0,.., x8, y8, conf, cls)
    """

    nc = prediction.shape[2] - 19  # number of classes
    xc = prediction[:, :, 18] > conf_thres  # candidates

    # Settings
    max_det = max_det  # maximum number of detections per image
    # max_nms = 1  
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1 and  multi_label # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 20))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference

        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 19:] *= x[:, 18:19]  # conf = obj_conf * cls_conf
        box = x[:, :18]


        # # elif n > max_nms:  # excess boxes
        if multi_label:
            i, j = (x[:, 19:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 19:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 19:20] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        # print(f"number of boxes {n}")
        if not n:  # no boxes
            continue
        elif n > max_det:
            x = x[x[:, 18].argsort(descending=True)[:max_det]]  # sort by confidence

        # print(f"final{x=}")
        output[xi] = x
        if (time.time() - t) > time_limit:
            print(f'WARNING: filter time limit {time_limit}s exceeded')
            break  # time limit exceeded
    
    return output

class MeshPly:
    def __init__(self, filename, color=[0., 0., 0.]):

        f = open(filename, 'r')
        self.vertices = []
        self.colors = []
        self.indices = []
        self.normals = []

        vertex_mode = False
        face_mode = False

        nb_vertices = 0
        nb_faces = 0

        idx = 0

        with f as open_file_object:
            for line in open_file_object:
                elements = line.split()
                if vertex_mode:
                    self.vertices.append([float(i) for i in elements[:3]])
                    self.normals.append([float(i) for i in elements[3:6]])

                    if elements[6:9]:
                        self.colors.append([float(i) / 255. for i in elements[6:9]])
                    else:
                        self.colors.append([float(i) / 255. for i in color])

                    idx += 1
                    if idx == nb_vertices:
                        vertex_mode = False
                        face_mode = True
                        idx = 0
                elif face_mode:
                    self.indices.append([float(i) for i in elements[1:4]])
                    idx += 1
                    if idx == nb_faces:
                        face_mode = False
                elif elements[0] == 'element':
                    if elements[1] == 'vertex':
                        nb_vertices = int(elements[2])
                    elif elements[1] == 'face':
                        nb_faces = int(elements[2])
                elif elements[0] == 'end_header':
                    vertex_mode = True