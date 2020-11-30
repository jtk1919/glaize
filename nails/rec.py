import os
import io
import sys
import numpy as np
import cv2
import argparse
import glob
import csv
import PIL
import configparser

import image_proc as ip

CONFIG_FILE = "C:\\Apps\\glaize\\glaize_config.txt"
if os.name == 'posix':
    CONFIG_FILE = "/home/taxila/cfg/glaize_config.txt"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)
DATA_DIR = config['DEFAULT']['data_dir']

IMAGES_DIR = DATA_DIR + 'images/left_fingers/'
LEFT_THUMB_DIR = DATA_DIR + 'images/left_thumb/'
RESULTS_DIR = DATA_DIR + 'results/left_fingers/'
TEST_DIR = DATA_DIR + 'test\\'
# IMAGES_DIR = "D:\\data\\fn\\"
# RESULTS_DIR = "D:\\data\\results\\fake_nails\\"


csvf = TEST_DIR + "rec.csv"
errf = TEST_DIR + "err.csv"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=0, help='model index, default 0')
parser.add_argument('--image_name', type=str, default='test_image1.jpg', help='name of image')
#parser.add_argument('--save_images', type=boolean, default=False, help='whether or not to save images (default False)')
opt = parser.parse_args()


ROOT_DIR = os.path.abspath("Mask_RCNN-master")

NAILS_MODEL_PATH = os.path.join( ROOT_DIR, "nails20201023T1913", "mask_rcnn_nails_{}.h5")


#sys.path.append(ROOT_DIR)
sys.path.insert( 0, ROOT_DIR)
import mrcnn.model as modellib
from mrcnn.config import Config
from image import *

MODEL_DIR = "D:/data/models/"

NAILS_MODEL = [ '0', '1', '2' ]
NAILS_MODEL[0] = os.path.join( MODEL_DIR, "mask_rcnn_nails_v1.h5")
NAILS_MODEL[1] = os.path.join( MODEL_DIR, "mask_rcnn_nails_t1.h5")
NAILS_MODEL[2] = os.path.join( MODEL_DIR, "mask_rcnn_nails_0026_t2.h5")

IMAGE_DIR = os.path.join(ROOT_DIR, "../nail_images")

try:
    os.mkdir(TEST_DIR)
except:
    print("{} exists".format(TEST_DIR))


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
##config.display()


def get_fnames_from_path(fpath):
    path, fname = os.path.split(fpath)
    fn, ext = fname.split(".")
    x = fpath.replace('left_fingers', 'left_thumb')
    x = x.replace('Left Fingers', 'Left Thumb')
    x = x.replace('Left fingers', 'Left thumb')
    x = x.replace('left fingers', 'left thumb')
    return fn, x, ext


def area_from_mask(mask):
    area_pixels_squared = np.sum(mask)
    area_mm_squared = area_pixels_squared / (image.pixels_per_mm ** 2)
    return area_mm_squared


def masked_image(img, mask):
    im = img.copy()
    for i in range(3):
        im[:, :, i][mask] = 0;
    return im


def denoise( bmask ):
    msk = np.zeros( bmask.shape[:2], dtype="uint8")
    msk[bmask] = 255
    contours, _ = cv2.findContours(msk, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    if len(contours) > 1:
        max_area = 0
        max_index = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > max_area):
                max_area = area
                max_index = i
        contour = contours[max_index]
    hull = cv2.convexHull(contour, False)
    mask = np.zeros(bmask.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, [hull], 0, 255, -1, 8)
    return ( mask > 0)



# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-NAILS
print( "Running inference with model ", NAILS_MODEL[ opt.model ], "\n" )
model.load_weights( NAILS_MODEL[ opt.model ], by_name=True)
class_names = ['BG', 'nail', 'card']
orientation = 0


def main( img, fn, is_left = True ):
    global model
    global orientation
    #
    rtn = False
    img1 = img.copy()
    results = model.detect([img], verbose=1)
    r = results[0]
    #
    n_regions = len(r['rois'][:, 0])
    if n_regions not in [2, 5]:
       print("Wrong number of regions ({}) detected!".format(len(r['rois'][:, 0])))
    #
    cc_i = 0
    max_area = 0
    areas = [0] * n_regions
    im2d = np.ones(img.shape[:2], np.uint8)
    for i in range(n_regions):
        area = np.sum(im2d[r['masks'][:, :, i]])
        areas[i] = area
        if area > max_area:
            max_area = area
            cc_i = i
    print( "credit card index: ", cc_i)
        #
    msk = np.zeros(img.shape[:2], np.uint8)
    for i in range(n_regions):
        if i != cc_i:
            bmask = denoise(r['masks'][:, :, i])
            msk[bmask] = 255
            img1 = masked_image(img1, bmask )
    if n_regions == 5:
        orientation, msk = ip.upright(msk)
        #
        rtn = False
    if orientation == 0:
        if n_regions == 2:
            _ = cv2.imwrite(TEST_DIR + fn + "_lthumb.png", img1)
            _ = cv2.imwrite(TEST_DIR + fn + "_lt_mask.png", msk)
            idx = 0
            if cc_i == idx:
                idx += 1
            clipped_nail, coord = ip.clip_finger_mask(idx, r, msk)
            _ = cv2.imwrite(TEST_DIR + fn + "_l4.png", clipped_nail)
            clipped_finger = ip.get_finger_clip( img1, coord)
            _ = cv2.imwrite(TEST_DIR + fn + "_fin4.png", clipped_finger)
            rtn = True
        elif n_regions == 5:
            _ = cv2.imwrite(TEST_DIR + fn + "_image.png", img)
            _ = cv2.imwrite(TEST_DIR + fn + "_nails.png", img1)
            _ = cv2.imwrite(TEST_DIR + fn + "_mask.png", msk)
            widths = []
            for i in [0, 1, 2, 3, 4]:
                if i != cc_i:
                    y1, x1, y2, x2 = r['rois'][i]
                    widths.append((i, x1))
            widths.sort(key=lambda x: x[1])
            for i in [0, 1, 2, 3]:
                idx = widths[i][0]
                clipped_nail, coord = ip.clip_finger_mask(idx, r, image)
                clipped_nail = cv2.cvtColor(clipped_nail, cv2.COLOR_GRAY2RGB)
                _ = cv2.imwrite(TEST_DIR + fn + "_l{}.png".format(i), clipped_nail)
                clipped_finger = ip.get_finger_clip(img1, coord)
                _ = cv2.imwrite(TEST_DIR + fn + "_fin{}.png".format(i), clipped_finger)
            rtn = True
        else:
            _ = cv2.imwrite(TEST_DIR + fn + "_image.png", img)
            _ = cv2.imwrite(TEST_DIR + fn + "_nails_err.png", img1)
            _ = cv2.imwrite(TEST_DIR + fn + "_mask_err.png", msk)
            print("Wrong number of regions ({}) detected!".format(len(r['rois'][:, 0])))
            rtn = False
    return rtn

##-----------------------------------------
c = 0
with open( csvf, 'w+', newline='') as csvfile:
    errfile = open( errf, 'w+', newline='')
    writer = csv.writer(csvfile)
    err_writer = csv.writer(errfile)
    files = glob.glob(IMAGES_DIR + "*")
    for f in files:
        c += 1
        fn, thf, ext = get_fnames_from_path(f)
        if not ( ext in ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG'] ):
            print( "File extension {} not accepted. Please provide as JPEG or PNG images. File: {}".format(ext, fn))
            continue
        print( "Running recognition on left hand: ", fn )
        #
        image = cv2.imread(f)
        img1 = image.copy()
        orientation = 0
        try:
            success = main( img1, fn )
            if not success:
                err_writer.writerow([f, thf, 'failed for fingers'])
                errfile.flush()
                continue
        except:
            err_writer.writerow([f, thf, 'failed for fingers'])
            errfile.flush()
            continue
        #
        lth = cv2.imread(thf)
        # turn the thumb according to the corresponding fingers orientation
        if orientation == 1:
            lth = cv2.rotate(lth, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 2 :
            lth = cv2.rotate(lth, cv2.ROTATE_180)
        elif orientation == 3:
            lth = cv2.rotate(lth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        try:
            print( "Thumb file size: ", lth.shape )
        except:
            print("The thumb file for left fingers [{}] does not exist at [{}]".format(f, lth))
            err_writer.writerow([f, thf, 'thumb file does not exist'])
            errfile.flush()
            continue
        try:
            success = main( lth, fn)
            if success:
                writer.writerow([ f, thf])
                csvfile.flush()
            else:
                err_writer.writerow([f, thf, 'failed for thumb'])
                errfile.flush()
        except:
            err_writer.writerow([f, thf, 'failed for fingers'])
            errfile.flush()

errfile.close()
print( "\nRCNN Recognition failed for the following hand images:")
_ = os.system( "type {}".format( errf ) )
print( "\nRCNN Recognition succeeded for the following hand images:")
_ = os.system( "type {}".format( csvf ) )