import os, io
import sys
import numpy as np
import cv2
import argparse
import glob
import math
import imutils
import pyheif
import PIL

import image_proc as ip



IMAGES_DIR = "D:\\data\\images\\left_fingers\\"
RESULTS_DIR = "D:\\data\\results\\left_fingers\\"
TEST_DIR = "D:\\data\\test\\"
#IMAGES_DIR = "D:\\data\\fn\\"
#RESULTS_DIR = "D:\\data\\results\\fake_nails\\"


parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='test_image1.jpg', help='name of image')
parser.add_argument('--save_images', type=str, default=False, help='whether or not to save images (default False)')
opt = parser.parse_args()

ROOT_DIR = os.path.abspath("Mask_RCNN-master")

sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn.config import Config
from image import *

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

NAILS_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_nails_v1.h5")

IMAGE_DIR = os.path.join(ROOT_DIR, "../nail_images")

class NailsConfig(Config):
    """
    Derives from the base Config class and overrides some values.
    """
    NAME = "nails"
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + nails + card
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

config = NailsConfig()


class InferenceConfig(config.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2
    DETECTION_MIN_CONFIDENCE = 0.8

config = InferenceConfig()
config.display()

def read_heic(path):
    with open(path, 'rb') as file:
        im = pyheif.read_heif(file)
        for metadata in im.metadata or []:
            if metadata['type'] == 'Exif':
                fstream = io.BytesIO(metadata['data'][6:])
    pi = PIL.Image.open(fstream)
    pi.save("temp.PNG", "PNG")
    im = cv2.imread( "temp.PNH")
    return im


def get_fname_from_path(fpath):
    path, fname = os.path.split(fpath)
    fn, ext = fname.split(".")
    return fn

def area_from_mask(mask):
    area_pixels_squared = np.sum(mask)
    area_mm_squared = area_pixels_squared/(image.pixels_per_mm**2)
    return area_mm_squared

def masked_image( img, mask):
    im = img.copy()
    for i in range(3):
        im[:,:,i][mask] = 0;
    return im

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-NAILS
model.load_weights(NAILS_MODEL_PATH, by_name=True)

class_names = ['BG', 'nail', 'card']


files = glob.glob( IMAGES_DIR + "*" )
for f in files:
    fn = get_fname_from_path(f)
    #
    image = cv2.imread(f)
    img = image.copy()
    img1 = image.copy()
    #
    results = model.detect([image], verbose=1)
    #
    r = results[0]
    #
    n_regions = len(r['rois'][:, 0])
    #
    cc_i = 0
    max_area = 0
    areas = [0] * n_regions
    im2d = np.ones( img.shape[:2])
    for i in range(n_regions):
        area = np.sum( im2d[ r['masks'][:, :, i] ] )
        areas[i] = area
        if area > max_area:
            max_area = area
            cc_i = i
    noise_regions = [cc_i]
    #
    ## noise removal
    #if n_regions > 5 :
    #    area_med = np.median(areas)
    #    for i in range(n_regions):
    #        if  area[i] < 0.2 * area_med :
    #            noise_regions.append[i]
    #
    rc = cv2.imwrite( TEST_DIR + fn + "_cc.png",  masked_image(img1, r['masks'][:,:,cc_i] ))
    im2d[r['masks'][:,:,cc_i]] = 255
    rc = cv2.imwrite( TEST_DIR + fn + "_cc_mask.png",  im2d)
    #
    msk = np.zeros( img.shape[:2])
    for i in range(n_regions):
        if i not in noise_regions:
            msk[r['masks'][:,:,i]] = 255
            img1 =  masked_image(img1, r['masks'][:,:,i] )
    #
    rc = cv2.imwrite( TEST_DIR + fn + "_nails.png",  img1)
    rc = cv2.imwrite( TEST_DIR + fn + "_nails_mask.png", msk)

##----------------------------------------------------------
## Hand orientation
msk1 = ip.upright( msk )




##----------------------------------------------------------
for i in range(n_regions):
    if i != cc_i:
        msk1 = np.zeros(img.shape[:2], dtype = "uint8")
        msk1[r['masks'][:, :, i]] = 1
        nail_clipped = ip.clip(msk1)
        contours, _ = cv2.findContours(nail_clipped, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ln = len(contours)
        if ln < 1:
            continue
        nail = contours[0]
        max_i = 0
        if ln > 1:
            max_area = 0
            for j in range( ln ):
                area = cv2.contourArea( contours[j] )
                if area > max_area:
                    max_area = area
                    max_i = j
        nail = contours[max_i]
        #
        bgr_nail = cv2.cvtColor( nail_clipped,cv2.COLOR_GRAY2BGR)
        angl = get_orientation( nail, bgr_nail)
        rotated = imutils.rotate_bound(bgr_nail, 90 - angl * 180 / math.pi)
        cv2.imwrite(TEST_DIR + fn + "_nail{}.png".format(i), bgr_nail)
        cv2.imwrite(TEST_DIR + fn + "_rotated{}.png".format(i), rotated)
        cv2.imshow("nail{}".format(i), bgr_nail)
        cv2.imshow("rotated{}".format(i), rotated)
        cv2.waitKey(0)