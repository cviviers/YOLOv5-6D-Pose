# Dataset utils and dataloaders
import matplotlib.pyplot as plt
import matplotlib
import glob
import logging
import math
import os, sys
import random
import shutil
import time
import json
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ExifTags, ImageMath, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm


from utils.general import clean_str
from utils.torch_utils import torch_distributed_zero_first
from utils.pose_utils import get_3D_corners, pnp, get_camera_intrinsic, compute_projection, MeshPly
from utils.occlude import load_occluders, occlude_with_objects

# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=True, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', bg_file_names=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabelsPose(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      bg_file_names=bg_file_names)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabelsPose.collate_fn4 if quad else LoadImagesAndLabelsPose.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')


        # Padded resize
        img, _, pad = letterbox(img0, self.img_size, stride=self.stride)

        h0 = img0.shape[0]
        w0 = img0.shape[1]
        h = img.shape[0]
        w = img.shape[1]

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, None, shapes

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, None

    def __len__(self):
        return 0

class LoadVideo: 
    def __init__(self, sources, camera_intrinsics, mesh_path, img_size=640, stride=32) -> None:
        pass
        self.img_size = img_size
        self.stride = stride
        with open(camera_intrinsics, 'r') as f:
            camera_data = json.load(f)

        # camera_model = None
        self.dtx = np.array(camera_data["distortion"])
        self.mtx = np.array(camera_data["intrinsic"])
        self.intrinsics = []
         
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            self.intrinsics[i] = list(self.mtx[0,0],  self.mtx[1,1], w, h, self.mtx[0,2], self.mtx[1,2], w, h)
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
        print('')  # newline

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, self.intrinics, None        
       

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

def Linemodimg2label_paths(img_paths):
    # Define label paths as a function of image paths
    return [x.replace("JPEGImages", "labels").replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt") for x in img_paths]

def Linemodimg2mask_paths(img_paths):
    # Define mask paths as a function of image paths
    if os.name == 'nt':
        return [x.replace("JPEGImages", "mask").replace("images", "mask").replace('\\00', '\\').replace(".jpg", ".png") for x in img_paths]
    else:
        return [x.replace("JPEGImages", "mask").replace("images", "mask").replace('/00', '/').replace(".jpg", ".png") for x in img_paths]

def Linemodimg2mask_path(img_paths):
    # Define mask paths as a function of image paths
    return img_paths.replace("JPEGImages", "mask").replace("images", "mask").replace('/00', '/').replace(".jpg", ".png")

class LoadImagesAndLabelsPose(Dataset):  # for training/testing
    def __init__(self, path, img_size=(640, 480), batch_size=16, augment=True, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', bg_file_names = None):

        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect =  rect

        self.stride = stride
        self.path = path
        self.bg_file_names = bg_file_names
        self.occluders = None

        if bg_file_names is not None and self.hyp['occlude'] > 0:
            
            occlude_path = os.path.join(bg_file_names[0].split("VOC2012")[0], 'VOC2012')
            print(f"Creating occluders from VOC: {occlude_path}")
            self.occluders = load_occluders(pascal_voc_root_path=occlude_path)

        
        data_dir = os.path.dirname(path)

        images_folder = None
        for folder in os.listdir(data_dir):
            if 'images' in folder.lower():
                images_folder = os.path.join(data_dir, folder)

        if images_folder is None:
            raise Exception(f'Error loading data from {data_dir}: Could not find images folder')
        # images_folder = os.path.join(data_dir, 'JPEGImages')
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([os.path.join(images_folder, x.replace('/', os.sep)) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix} No images found'
        except Exception as e:
            raise Exception(f'{prefix} Error loading data from {path}: {e}')

        # Check cache
        self.label_files = Linemodimg2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        self.mask_files = Linemodimg2mask_paths(self.img_files)  # mask
        if cache_path.is_file():
            
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.mask_files + self.img_files) or 'version' not in cache:  # changed
                cache, exists = self.cache_data(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_data(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, nm_mask, nf_mask, ne_mask, nc_mask , n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' for images, labels, masks... {nf} found, {nm} missing, {ne} empty, {nc} corrupted, "\
            f"{nf_mask} found, {nm_mask} missing, {ne_mask} empty, {nc_mask} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels.'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version

        labels, masks, shapes = zip(*cache.values())

        self.labels = list(labels)
        self.masks = list(masks)
        self.shapes = np.array(shapes, dtype=np.float64)
 
        self.img_files = list(cache.keys())  # update
        self.label_files = Linemodimg2label_paths(cache.keys())  # update

        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.masks = [self.masks[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'

    def cache_data(self, path=Path('./data.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, nm_mask, nf_mask, ne_mask, nc_mask = 0, 0, 0, 0, 0, 0, 0, 0 # number missing, found, empty, duplicate, number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files, self.mask_files), desc='Scanning images', total=len(self.img_files))

        for i, (im_file, lb_file, mk_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] >= 27, 'labels at least require 21 columns each + 6 camera intrinsics'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 27), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 27), dtype=np.float32)
                    # raise NameError('label missing')
                
                 # verify masks
                if os.path.isfile(mk_file):
                    nf_mask += 1  # mask found
                    mk = Image.open(mk_file)
                    mk.verify()
                    if exif_size(mk) != shape:
                        mk = np.zeros(shape, dtype=np.float32)
                        assert exif_size(mk) == shape, 'mask does not fit image'
                else:
                    nm_mask += 1  # mask missing
                    mk = None
                    mk_file = None


                x[im_file] = [l, mk_file, shape]
            except Exception as e:
                nc_mask += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' for images, labels and masks... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted, "\
                        f"{nf_mask} found, {nm_mask} missing, {ne_mask} empty, {nc_mask} corrupted"

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}.')
        if nf_mask == 0:
            print(f'{prefix}WARNING: No masks found in {path}.')

        x['hash'] = get_hash(self.label_files + self.mask_files + self.img_files)
        x['results'] = nf, nm, ne, nc, nm_mask, nf_mask, ne_mask, nc_mask, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x
    
    
    
    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)
        
        img2 = img

        # Augment background
        if self.augment: 
            mask = cv2.imread(self.masks[index])
            if hyp['background'] and self.bg_file_names is not None and self.masks[index] != None:

                    if random.random() < hyp['background']:
                        # Get background image path
                        random_bg_index = random.randint(0, len(self.bg_file_names) - 1)
                        bgpath = self.bg_file_names[random_bg_index]
                        bg = cv2.imread(bgpath)
                        img = change_background(img, mask, bg)

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        nl = len(labels)  # number of labels
        labels_og = labels.copy()

        if labels.size:  # normalized xy to pixel xy format
            labels[:, 1:19] = xy_norm2xy_pix (labels[:, 1:19], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            # labels[:, 19:21] = compute_new_width_height(labels[:, 1:19])

        if self.augment:

            #Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 1:19][:, ::2] = 1 - labels[:, 1:19][:, ::2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1:19][:, 1::2] = 1 - labels[:, 1:19][:, 1::2]
            
            img, labels = random_pose_perspective(img, labels, degrees=hyp['degrees'], translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            if hyp['occlude'] and self.bg_file_names is not None and self.masks[index] != None:

                if random.random() < hyp['occlude'] and self.occluders:
                    img = occlude_with_objects(img, self.occluders)

            # Augment blur
            if random.random() < hyp['blur']:
                img = gaussian_blur(img, hyp['blur_kernel'])

               
        if labels.size:  #  pixel xyxy format to normalized xy
            labels[:, 1:21] = xy_pix2xy_norm(labels[:, 1:21], img.shape[1], img.shape[0], padw=0, padh=0)
            labels[:, 19:21] = compute_new_width_height(labels[:, 1:19]) 

        if False:

            corners2D_gt = np.array(np.reshape(labels_og[0, 1:9*2+1], [9, 2]), dtype='float32')
            corners2D_gt[:, 0] = corners2D_gt[:, 0] * w
            corners2D_gt[:, 1] = corners2D_gt[:, 1] * h 
            corners2D_gt_aug = np.array(np.reshape(labels[0, 1:9*2+1], [9, 2]), dtype='float32')
            corners2D_gt_aug[:, 0] = corners2D_gt_aug[:, 0] * img.shape[1]
            corners2D_gt_aug[:, 1] = corners2D_gt_aug[:, 1] * img.shape[0] 

            corn2D_gt= corners2D_gt[1:, :]
            corn2D_gt_aug= corners2D_gt_aug[1:, :]

            edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
            # # # Visualize
            
            # matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(img2[:, :, ::-1])  # base
            ax[0].scatter(corners2D_gt.T[0], corners2D_gt.T[1], c='r', s = 20)
            
            ax[1].imshow(img[:, :, ::-1])  # warped
            ax[1].scatter(corners2D_gt_aug.T[0], corners2D_gt_aug.T[1], c='r', s = 20)
            # Projections
            for edge in edges_corners:
                ax[1].plot(corn2D_gt_aug[edge, 0], corn2D_gt_aug[edge, 1], color='g', linewidth=1.0)
                ax[0].plot(corn2D_gt[edge, 0], corn2D_gt[edge, 1], color='g', linewidth=1.0)
            plt.show()
            plt.savefig('test.png')


        labels_out = torch.zeros((nl, 22))
        intrinics = torch.zeros((nl, 6))

        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels[:, :21])
            intrinics[:, :] = torch.from_numpy(labels[:, 21:27])
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, intrinics, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, intrinics, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
            # intrinics[i][:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), torch.cat(intrinics, 0), path, shapes

def xy_norm2xy_pix(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, ::2] = w * (x[:, ::2]) + padw 
    y[:, 1::2] = h * (x[:, 1::2]) + padh  
    return y

def xy_pix2xy_norm(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, ::2] =  (x[:, ::2] - padw)/w 
    y[:, 1::2] = (x[:, 1::2]-padh)/h 
    return y

def gaussian_blur(cv_image, kernel_size = 5):
    return cv2.GaussianBlur(cv_image, (kernel_size,kernel_size), cv2.BORDER_DEFAULT)

# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

def resize_mask(msk, img_size, augment):

    h0, w0 = msk.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
        msk = cv2.resize(msk, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return msk
    

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 480), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)



def change_background(img, mask, bg):

    # ow, oh = img.shape[:2]
    
    bg = cv2.resize(bg, img.shape[:2][::-1]) # [:, :, ::-1]
    mask = cv2.resize(mask, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    #bg = np.array(bg)
    # mask = mask
    if mask.ndim == 2:
        mask = np.stack((mask,mask,mask), axis=2)

    new_img = np.where(mask[:, :, :] ==  True, img, bg)

    return new_img

def random_pose_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):

    # targets = [cls, xy*9, label_width, label_height]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)

    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 9 , 3))
        xy[:, :2] = targets[:, 1:19].reshape(n * 9, 2)  # x1y1, x2y2, x3y3, x4y4, ...., x9y9
        xy = xy @ M.T # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 18)  # perspective rescale or affine
        targets[:, 1:19] = xy
        targets[:, 19:21] = compute_new_width_height(xy)

    return img, targets

def augmentation_6DoF(img, mask, rotation_matrix_annos, translation_vector_annos, angle, scale, camera_matrix, mask_values):

    """ Computes the 6D augmentation.
    Args:
        img: The image to augment
        mask: The segmentation mask of the image
        rotation_matrix_annos: numpy array with shape (num_annotations, 3, 3) which contains the ground truth rotation matrix for each annotated object in the image
        translation_vector_annos: numpy array with shape (num_annotations, 3) which contains the ground truth translation vectors for each annotated object in the image
        angle: rotate the image with the given angle
        scale: scale the image with the given scale
        camera_matrix: The camera matrix of the example
        mask_values: numpy array of shape (num_annotations,) containing the segmentation mask value of each annotated object
    Returns:
        augmented_img: The augmented image
        augmented_rotation_vector_annos: numpy array with shape (num_annotations, 3) which contains the augmented ground truth rotation vectors for each annotated object in the image
        augmented_translation_vector_annos: numpy array with shape (num_annotations, 3) which contains the augmented ground truth translation vectors for each annotated object in the image
        augmented_bbox_annos: numpy array with shape (num_annotations, 4) which contains the augmented ground truth 2D bounding boxes for each annotated object in the image
        still_valid_annos: numpy boolean array of shape (num_annotations,) indicating if the augmented annotation of each object is still valid or not (object rotated out of the image for example)
        is_valid_augmentation: Boolean indicating wheter there is at least one valid annotated object after the augmentation
    """

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    height, width, _ = img.shape
    #rotate and scale image

    a = random.uniform(-angle, angle) if angle != 0 else 0.0
    s = random.uniform(1 - scale, 1 + scale) if scale != 0 else 0.0
    # print(a, s)
    rot_2d_mat = cv2.getRotationMatrix2D((cx, cy), -a, s)

    augmented_img = cv2.warpAffine(img, rot_2d_mat, (width, height))
    #append the affine transformation also to the mask to extract the augmented bbox afterwards
    augmented_mask = cv2.warpAffine(mask, rot_2d_mat, (width, height), flags = cv2.INTER_NEAREST) #use nearest neighbor interpolation to keep valid mask values
    num_annos = rotation_matrix_annos.shape[0]

    augmented_rotation_matrix_annos = np.zeros((num_annos, 3, 3), dtype = np.float32)
    augmented_translation_vector_annos = np.zeros((num_annos, 3), dtype = np.float32)
    augmented_bbox_annos = np.zeros((num_annos, 4), dtype = np.float32)

    still_valid_annos = np.zeros((num_annos,), dtype = bool) #flag for the annotations if they are still in the image and usable after augmentation or not
    for i in range(num_annos):
        augmented_bbox, is_valid_augmentation = get_bbox_from_mask(augmented_mask, mask_value = mask_values[i])
        
        if not is_valid_augmentation:
            still_valid_annos[i] = False
            continue
        #create additional rotation vector representing the rotation of the given angle around the z-axis in the camera coordinate system
        tmp_rotation_vector = np.zeros((3,))
        tmp_rotation_vector[2] = a / 180. * math.pi
        tmp_rotation_matrix, _ = cv2.Rodrigues(tmp_rotation_vector)
        #get the final augmentation rotation
        augmented_rotation_matrix = np.dot(tmp_rotation_matrix, rotation_matrix_annos[i, :, :])
        # augmented_rotation_vector, _ = cv2.Rodrigues(augmented_rotation_matrix)
        
        #also rotate the gt translation vector first and then adjust Tz with the given augmentation scale
        augmented_translation_vector = np.dot(np.copy(translation_vector_annos[i]), tmp_rotation_matrix.T)
        augmented_translation_vector[ 2] /= s
        
        #fill in augmented annotations
        augmented_rotation_matrix_annos[i, :, :] = augmented_rotation_matrix # np.squeeze(augmented_rotation_vector)
        augmented_translation_vector_annos[i, :] = np.squeeze(augmented_translation_vector)
        augmented_bbox_annos[i, :] = augmented_bbox
        still_valid_annos[i] = True
        
    return augmented_img, augmented_rotation_matrix_annos, augmented_translation_vector_annos, augmented_bbox_annos, still_valid_annos, True

def get_bbox_from_mask( mask, mask_value = None):
    """ Computes the 2D bounding box from the input mask
    Args:
        mask: The segmentation mask of the image
        mask_value: The integer value of the object in the segmentation mask
    Returns:
        numpy array with shape (4,) containing the 2D bounding box
        Boolean indicating if the object is found in the given mask or not
    """
    if mask_value is None:
        seg = np.where(mask != 0)
    else:
        seg = np.where(mask == mask_value)
    #check if mask is empty
    if seg[0].size <= 0 or seg[1].size <= 0:
        return np.zeros((4,), dtype = np.float32), False
    min_x = np.min(seg[1])
    min_y = np.min(seg[0])
    max_x = np.max(seg[1])
    max_y = np.max(seg[0])
    
    return np.array([min_x, min_y, max_x, max_y], dtype = np.float32), True

def compute_new_width_height(coordinates):
    return np.array([[np.max(coordinates[:, ::2])-np.min(coordinates[:, ::2]), np.max(coordinates[:, 1::2])-np.min(coordinates[:, 1::2])]])


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)
