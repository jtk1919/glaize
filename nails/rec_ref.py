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
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
DATA_DIR = config['DEFAULT']['data_dir']

IMAGES_DIR = DATA_DIR + 'ref_nails\\'
LEFT_THUMB_DIR = DATA_DIR + 'ref_nails\\'
RESULTS_DIR = DATA_DIR + 'results\ref_fingers\\'
TEST_DIR = DATA_DIR + 'testref\\'

REF_FINGERS = [ 'Left fingers combi 1', 'Left fingers combi 2 and 3',
                'Left fingers combi 4', 'Left fingers combi 5 and 7',
                'left fingers combi 6', 'Left fingers combi 8 and 9',
                'Left fingers combi 10','Left fingers combi 11',
                'Left fingers combi 12', 'Left fingers combi 13 and 14']
REF_THUMBS = [ 'Thumb combi 1 and 2', 'Thumb combi 3, 4 and 5',
               'Thumb combi 6, 7 and 8', 'Thumb combi 9, 10, 11, 12 and 13',
               'Thumb combi 14']



csvf = TEST_DIR + "rec.csv"
csverr = TEST_DIR + "err.csv"

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


def createDirs():
    try:
        os.mkdir( TEST_DIR )
    except:
        print( "{} exists".format( TEST_DIR))
    os.chdir(TEST_DIR)
    try:
        for i in range(len(REF_FINGERS)):
            os.mkdir( REF_FINGERS[i] )
    except:
        print( "Error while creating finger sub-directories in ", TEST_DIR)
    try:
        for i in range(len(REF_THUMBS)):
            os.mkdir( REF_THUMBS[i] )
    except:
        print( "Error while creating thumb sub-directories in ", TEST_DIR)



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
    im = cv2.imread("temp.PNH")
    return im


def get_fnames_from_path(fpath):
    path, fname = os.path.split(fpath)
    fn, ext = fname.split(".")
    write_path = path.replace( 'ref_nails', 'testref') + '\\'
    path = path + '\\'
    return fn, path, write_path


def area_from_mask(mask):
    area_pixels_squared = np.sum(mask)
    area_mm_squared = area_pixels_squared / (image.pixels_per_mm ** 2)
    return area_mm_squared


def masked_image(img, mask):
    im = img.copy()
    for i in range(3):
        im[:, :, i][mask] = 0;
    return im


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-NAILS
model.load_weights(NAILS_MODEL_PATH, by_name=True)
class_names = ['BG', 'nail', 'card']
orientation = 0


def main( img, fn, write_path, is_left = True ):
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
    im2d = np.ones(img.shape[:2])
    for i in range(n_regions):
        area = np.sum(im2d[r['masks'][:, :, i]])
        areas[i] = area
        if area > max_area:
            max_area = area
            cc_i = i
    print( "credit card index: ", cc_i)
    #
    msk = np.zeros(img.shape[:2])
    for i in range(n_regions):
        if i != cc_i:
            msk[r['masks'][:, :, i]] = 255
            img1 = masked_image(img1, r['masks'][:, :, i])
    #
    ##if n_regions == 5:
    ##    orientation, msk = ip.upright(msk)
    orientation = 0
    if orientation == 0:
        if n_regions == 2:
            _ = cv2.imwrite(write_path + fn + "_lthumb.png", img1)
            _ = cv2.imwrite(write_path + fn + "_lt_mask.png", msk)
            idx = 0
            if cc_i == idx:
                idx += 1
            clipped_nail = ip.clip_finger_mask(idx, r, msk)
            _ = cv2.imwrite(write_path + fn + "_l4.png", clipped_nail)
            rtn = True
        elif n_regions == 5:
            _ = cv2.imwrite(write_path + fn + "_image.png", img)
            _ = cv2.imwrite(write_path + fn + "_nails.png", img1)
            _ = cv2.imwrite(write_path + fn + "_mask.png", msk)
            widths = []
            for i in [0, 1, 2, 3, 4]:
                if i != cc_i:
                    y1, x1, y2, x2 = r['rois'][i]
                    widths.append((i, x1))
            widths.sort(key=lambda x: x[1])
            for i in [0, 1, 2, 3]:
                idx = widths[i][0]
                clipped_nail = ip.clip_finger_mask(idx, r, image)
                clipped_nail = cv2.cvtColor(clipped_nail, cv2.COLOR_GRAY2RGB)
                _ = cv2.imwrite(write_path + fn + "_l{}.png".format(i), clipped_nail)
            rtn = True
        else:
            _ = cv2.imwrite(write_path + fn + "_image.png", img)
            _ = cv2.imwrite(write_path + fn + "_nails_err.png", img1)
            _ = cv2.imwrite(write_path + fn + "_mask_err.png", msk)
            print("Wrong number of regions ({}) detected!".format(len(r['rois'][:, 0])))
            rtn = False
    return rtn, ( n_regions - 1 )

##-----------------------------------------
createDirs()
c = 0
e = 0
with open( csvf, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    errf = open( csverr, 'w', newline='')
    errwriter = csv.writer(errf)
    files = glob.glob(IMAGES_DIR + "*/*.png")
    for f in files:
        fn, pth, write_path = get_fnames_from_path(f)
        print( "Running recognition on: ", fn , " in ", pth )
        #
        image = cv2.imread(f)
        img1 = image.copy()
        success, nfingers = main( img1, fn, write_path )
        pth.replace( "\\", "\\\\")
        if success:
            writer.writerow([ nfingers, pth, fn + '.png'])
            csvfile.flush()
            c += 1
        else:
            errwriter.writerow( [ nfingers, pth, fn + '.png'] )
            errf.flush()
            e += 1
