import numpy as np
import cv2
from cv2 import aruco as aruco
import json
from json import JSONEncoder
from scipy import linalg
from scipy import spatial
from matplotlib.path import Path as mat_Path
import os
from PIL import Image, ImageDraw, ImageFont

def calc_rot_matrix(x_rot, y_rot, z_rot):
    X_rotation = x_rot * np.pi/180

    R_X = np.array([  [1, 0, 0],
                [0, np.cos(X_rotation), -1 * np.sin(X_rotation)],
                [0, np.sin(X_rotation), np.cos(X_rotation)]], dtype='float32')
    
    Y_rotation = y_rot *  np.pi/180
    R_Y = np.array([  [np.cos(Y_rotation), 0, np.sin(Y_rotation)],
                        [0, 1, 0],
                        [-1 * np.sin(Y_rotation), 0 , np.cos(Y_rotation)]], dtype='float32')

    Z_rotation = z_rot * np.pi/180
    R_Z = np.array([  [np.cos(Z_rotation), -1 * np.sin(Z_rotation), 0],
                        [np.sin(Z_rotation), np.cos(Z_rotation), 0],
                        [0, 0 , 1]], dtype='float32')

    return (R_X@(R_Y ))@(R_Z) 

def calc_rot_matrix_rad(x_rot, y_rot, z_rot):
    X_rotation = x_rot

    R_X = np.array([  [1, 0, 0],
                [0, np.cos(X_rotation), -1 * np.sin(X_rotation)],
                [0, np.sin(X_rotation), np.cos(X_rotation)]], dtype='float32')
    
    Y_rotation = y_rot
    R_Y = np.array([  [np.cos(Y_rotation), 0, np.sin(Y_rotation)],
                        [0, 1, 0],
                        [-1 * np.sin(Y_rotation), 0 , np.cos(Y_rotation)]], dtype='float32')

    Z_rotation = z_rot 
    R_Z = np.array([  [np.cos(Z_rotation), -1 * np.sin(Z_rotation), 0],
                        [np.sin(Z_rotation), np.cos(Z_rotation), 0],
                        [0, 0 , 1]], dtype='float32')
    # 1 (R_Z@(R_X ))@(R_Y)
    # 2 (R_Z@(R_Y ))@(R_X)
    # 3 (R_X@(R_Z ))@(R_Y)
    # 4 (R_X@(R_Y ))@(R_Z) maybe this
    # 5 (R_Y@(R_Z ))@(R_X)
    # 6 (R_Y@(R_X ))@(R_Z)
    return (R_X@(R_Y ))@(R_Z) 

def convert_quaternoins_to_3x3(arr):
    x, y, z, w = arr[0], arr[1], arr[2], arr[3]
    
    new_arr = np.array(((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), 
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w) ), 
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))) )

    return new_arr


def calculate_focal_length(system_focal_length, image_with=None, image_height=None, detector_width=None, detector_height=None):
    focal_x = image_with/detector_width*system_focal_length
    focal_y = image_height/detector_height*system_focal_length
    return focal_x, focal_y

def calc_focal_point(offset_center_x = 320, offset_center_y = 240, deviation_projected_center_x = 0, deviation_projected_center_y = 0):
    return offset_center_x +deviation_projected_center_x, offset_center_y+deviation_projected_center_y

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

def inverse_tranformation_matrix(matrix):
    rotation = matrix[:3, :3]
    translation = np.expand_dims(matrix[:3, 3], 1)
    inv_rot = np.linalg.inv(rotation)
    return np.vstack((np.hstack((inv_rot, -inv_rot@translation)), np.array((0, 0, 0, 1))))

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

def construct_transform(position, orientation):
    position = np.expand_dims(np.array(position), 1)
    normal_orientation = calc_rot_matrix(orientation[0], orientation[1], orientation[2])
    return np.vstack((np.hstack((normal_orientation, position)), np.array((0, 0, 0, 1))))

def draw(img, corners, imgpts):

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)

    return img


def drawBoxes(img, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def drawBoxes_refined(img, imgpts):
    edges_corners       = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    # imgpts = np.int32(imgpts).reshape(-1,2)
    for edge in edges_corners:
        img = cv2.line(img, tuple(imgpts[edge[0]]), tuple(imgpts[edge[1]]), (255), 1)

    return img

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    # return Vh[3,0:3]/Vh[3,3]

    X = Vh[3,0:3]/Vh[3,3]
    # X1 = P1 @ X
    # X2 = P2 @ X
    return X 

def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def create_mask(vertices, width, height, threshold):

    hull = spatial.ConvexHull(points=vertices)
    print(hull.vertices)
    mask = np.zeros((height, width), dtype=np.uint8)
    # cv2.fillConvexPoly(mask, hull, 255)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    print(f"{vertices[hull.vertices]=}")
    area_of_hull = PolyArea2D(vertices[hull.vertices])
    print(area_of_hull)
    mask = mat_Path(vertices[hull.vertices]).contains_points(np.vstack((x,y)).T)
    mask = mask.reshape((height, width)).astype(int)

    mask_area = cv2.countNonZero(mask)


    print(mask_area)
    return mask, mask_area/area_of_hull > threshold

def create_simple_mask(vertices, width, height):
    print(vertices.shape)
    hull = cv2.convexHull(vertices)
    hull = np.array(hull, dtype=np.int32)

    print(hull.shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask

def create_complex_mask(vertices, width, height, threshold):
    image_mask = np.zeros((height, width), dtype = np.uint8)
    for pixel in vertices:
        cv2.circle(image_mask, (int(pixel[0]),int(pixel[1])), 5, 255, -1)

    thresh = cv2.threshold(image_mask, 30, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key=cv2.contourArea)
    image_mask = np.zeros((height, width), dtype = np.uint8)
    cv2.drawContours(image_mask, [cnt], -1, 255, -1)

    mask_area = cv2.countNonZero(image_mask)

    return image_mask, True, mask_area
def dilate(img):
    dilatation_size = 1 # cv2.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = cv2.MORPH_RECT # 5#

    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilated_img = cv2.dilate(img, element)
    return dilated_img
    
def mask_by_threshold(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    (T, threshInv) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("Threshold Binary Inverse.png", threshInv)
    mask_area = cv2.countNonZero(threshInv)
    return threshInv, mask_area

def save_data(img, mask, labels, name , output_dir, projection):

    # Save projections.
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    projections_dir = os.path.join(output_dir, 'projections')
    if not os.path.isdir(projections_dir):
        os.mkdir(projections_dir)

    mask_dir = os.path.join(output_dir, 'mask')
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)

    labels_dir = os.path.join(output_dir, 'labels')
    if not os.path.isdir(labels_dir):
        os.mkdir(labels_dir)

    
    filename = os.path.join(images_dir, name)
    if img.dtype == np.uint16:
        img = (img/16).astype('uint8')
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
    img.save(filename) 

    filename = os.path.join(projections_dir, name)
    projection_image = Image.fromarray(cv2.cvtColor(projection, cv2.COLOR_BGR2RGB) )
    projection_image.save(filename)

    filename = os.path.join(mask_dir, name)
    masks = Image.fromarray(mask, 'L')
    masks.save(filename) 

    filename = os.path.join(labels_dir,  name[:-4]) + '.txt'

    with open(filename, 'w') as f:
        f.write(' '.join(map(str, labels)) )

def create_label(class_label, vertecies, focal_x, focal_y, sensor_width, sensor_height, x_offset, y_offset, im_width, im_height, transformation):
    
    label = []

    label.append(class_label)
   
    keypoints = vertecies.flatten('F').tolist() # flatten the array in column major order x0, y0, x1, y1, x2, y2, x3, y3....
   
    keypoints[0::2] = [float(i)/float(im_width) for i in keypoints[0::2]] # normalize the keypoints
    keypoints[1::2] = [float(i)/float(im_height) for i in keypoints[1::2]] # normalize the keypoints

    label.extend(keypoints)
    label.append(np.amax(keypoints[0::2], axis=0) - np.amin(keypoints[0::2], axis=0))
    label.append(np.amax(keypoints[1::2], axis=0) - np.amin(keypoints[1::2], axis=0))
    label.append(focal_x)
    label.append(focal_y)
    label.append(sensor_width)
    label.append(sensor_height)
    label.append(x_offset)
    label.append(y_offset)
    label.append(im_width)
    label.append(im_height)
    label.extend(cv2.Rodrigues(transformation[:3, :3])[0].T[0])
    label.extend(transformation[:3, 3].T)

    return label

def resize_projection(projection, x_ratio, y_ratio):
    return np.array([projection[:, 0]/x_ratio, projection[:, 1]/y_ratio]).T


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


# Detects the pose of the complete board
# this can be more accurate/robust than detecting the single markers since it allows for some markers to be occluded
def detect_Charuco_pose_board(img, matrix_coefficients, distortion_coefficients, markerSize = 6, totalMarkers = 50, 
                            number_markers_x = 8, number_markers_y = 5, square_width = 0.05, aruco_width = 0.03):
    

    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    img_copied = img.copy()
    arucoDict = aruco.getPredefinedDictionary(key)
    # arucoParam = aruco.DetectorParameters_create()
    arucoParam =  aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParam)

    board = aruco.CharucoBoard((number_markers_x, number_markers_y), square_width, aruco_width, arucoDict) # 8x5 sqaures, aruco marker is 3 centimeter, full square is 5 centimeter
    
    # charucoParam = aruco.CharucoParameters(cameraMatrix=matrix_coefficients, distCoeffs=distortion_coefficients)
    # refinmendParams = aruco.RefineParameters()
    charucoDetector = aruco.CharucoDetector(board =board,  detectorParams=arucoParam)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # board.setLegacyPattern(True) 

    pose = None
    frame_remapped_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(frame_remapped_gray)
    print("Number of markers detected: ", len(markerCorners))
    
    
    #corners, ids, rejected_img_points = detector.detectMarkers(frame_remapped_gray)
    # corners = refine_corners(frame_remapped_gray, corners)
    # print("Number of markers detected: ", len(corners))

    # if distortion_coefficients is not None:
    #     aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejected_img_points, cameraMatrix=matrix_coefficients, distCoeffs=distortion_coefficients)
    # else:
    #     aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejected_img_points, cameraMatrix=matrix_coefficients)
    
    if charucoIds is not None and charucoIds.size > 0: # if there is at least one marker detected

        # charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco( corners, ids, frame_remapped_gray, board)
        corners = cv2.cornerSubPix(frame_remapped_gray, charucoCorners, (11,11), (-1,-1), criteria)
        im_with_charuco_board = aruco.drawDetectedCornersCharuco(img_copied, charucoCorners, charucoIds, (0,255,0))
        im_with_charuco_board = aruco.drawDetectedMarkers(im_with_charuco_board, markerCorners, markerIds, (0,255,0))

        rvec, tvec = None, None
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, matrix_coefficients, distortion_coefficients, rvec= rvec, tvec=tvec)
            
        if retval == True:
            im_with_charuco_board = cv2.drawFrameAxes(im_with_charuco_board, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)  # axis length 100 can be changed according to your requirement
            pose = ({"rvec": rvec,  "tvec": tvec})
            return im_with_charuco_board, pose
        else:
            print("Could not retrieve pose from board")
            return im_with_charuco_board, None
    else:
        print("No markers detected")
        return img, None
    
def draw_BBox(img, corners, vertices, color = (255, 255, 0), thickness = 2):
    edges_corners       = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    colormap_in_rgb    = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0,255,255], [255, 0, 255], [255, 255, 0], [0, 0, 0], [255, 255, 255], [135, 206, 235]]
    
    # Draw lines between points
    for edge in edges_corners:
        start = tuple([round(corners[edge, 0][0]), round(corners[edge, 1][0])])
        end = tuple([round(corners[edge, 0][1]), round(corners[edge, 1][1])])
        cv2.line(img, start, end, color=color, thickness= thickness)
    # Add nice circles to the points
    for i, vertex in enumerate(vertices):
        vertex = tuple([round(vertex[0]), round(vertex[1])])
        cv2.circle(img, vertex, 3, color=tuple(colormap_in_rgb[i]), thickness=-1)
    
    return img