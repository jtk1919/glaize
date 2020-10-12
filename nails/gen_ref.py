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
                'Left fingers combi 6', 'Left fingers combi 8 and 9',
                'Left fingers combi 10','Left fingers combi 11',
                'Left fingers combi 12', 'Left fingers combi 13 and 14']
REF_THUMBS = [ 'Thumb combi 1 and 2', 'Thumb combi 3 4 and 5',
               'Thumb combi 6 7 and 8', 'Thumb combi 9 10 11 12 and 13',
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

ip.
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



##-----------------------------------------
csvfile = open( csvf, 'w', newline='')
writer = csv.writer(csvfile)
files = glob.glob(TEST_DIR + "Thumb*/*lthumb1.png")
for f in files:
    fn, pth, write_path = get_fnames_from_path(f)
    print( "Running reference nail processing on: ", fn , " in ", pth )
    #
    image = cv2.imread(f)
    img1 = image.copy()
    gray = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY_INV )
    r, c = mask.shape
    if c > r :
        mask = cv2.rotate( mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    msk1 = ip.nail_upright(mask)
    msk1 = cv2.cvtColor( msk1, cv2.COLOR_GRAY2BGR)
    _ = cv2.imwrite( pth + fn + "_mask.png", msk1)
    mask = ip.clip(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _ = cv2.imwrite(pth + fn + "_mask0.png", mask)
    writer.writerow([1, pth + fn + '.png'])
    csvfile.flush()

files = glob.glob(TEST_DIR + "Left fingers*/*_f?.png")
for f in files:
    fn, pth, write_path = get_fnames_from_path(f)
    print( "Running reference nail processing on: ", fn , " in ", pth )
    #
    image = cv2.imread(f)
    img1 = image.copy()
    gray = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY_INV )
    r, c = mask.shape
    if c > r :
        mask = cv2.rotate( mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    msk1 = ip.nail_upright(mask)
    msk1 = cv2.cvtColor( msk1, cv2.COLOR_GRAY2BGR)
    fnx = fn.replace( '_f', '_l')
    _ = cv2.imwrite( pth + fnx + ".png", msk1)
    mask = ip.clip(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _ = cv2.imwrite(pth + fnx + "_0.png", mask)


csvfile.close()
