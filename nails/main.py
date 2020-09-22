import os
import sys
import cv2
import argparse

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

def area_from_mask(mask):
    area_pixels_squared = np.sum(mask)
    area_mm_squared = area_pixels_squared/(image.pixels_per_mm**2)
    return area_mm_squared

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-NAILS
model.load_weights(NAILS_MODEL_PATH, by_name=True)

class_names = ['BG', 'nail', 'card']

image = cv2.imread(os.path.join(IMAGE_DIR, opt.image_name))

results = model.detect([image], verbose=1)

r = results[0]

n_regions = len(r['rois'][:, 0])
if (n_regions == 2):
    region1_H_begin, region1_W_begin, region1_H_end, region1_W_end = r['rois'][0]
    region2_H_begin, region2_W_begin, region2_H_end, region2_W_end = r['rois'][1]
    widths = [region1_W_begin, region2_W_begin]
    sort_indices = np.argsort(widths)
    finger1_index = sort_indices[1]
elif (n_regions == 5):
    region1_H_begin, region1_W_begin, region1_H_end, region1_W_end = r['rois'][0]
    region2_H_begin, region2_W_begin, region2_H_end, region2_W_end = r['rois'][1]
    region3_H_begin, region3_W_begin, region3_H_end, region3_W_end = r['rois'][2]
    region4_H_begin, region4_W_begin, region4_H_end, region4_W_end = r['rois'][3]
    region5_H_begin, region5_W_begin, region5_H_end, region5_W_end = r['rois'][4]
    widths = [region1_W_begin, region2_W_begin, region3_W_begin, region4_W_begin, region5_W_begin]
    sort_indices = np.argsort(widths)
    finger1_index = sort_indices[1]
    finger2_index = sort_indices[2]
    finger3_index = sort_indices[3]
    finger4_index = sort_indices[4]
else:
    raise Exception("Wrong number of regions ({}) detected!".format(len(r['rois'][:, 0])))


print("...masked predicted...")
print("Determining scale...")

image = IMAGE(path_to_image=IMAGE_DIR, name_of_image=opt.image_name, save_images=opt.save_images, card_width_mm=85.60)
line_distance_pixels = image.point_of_scale()

print("{} pixels per mm in the image".format(image.pixels_per_mm))

segmented_image = 255*np.ones(image.image.shape[:2])
if (n_regions == 2):
    area1 = area_from_mask(r['masks'][:,:,finger1_index])
    segmented_image[r['masks'][:, :, finger1_index]] = 0
elif (n_regions == 5):
    area1 = area_from_mask(r['masks'][:, :, finger1_index])
    area2 = area_from_mask(r['masks'][:, :, finger2_index])
    area3 = area_from_mask(r['masks'][:, :, finger3_index])
    area4 = area_from_mask(r['masks'][:, :, finger4_index])
    segmented_image[r['masks'][:, :, finger1_index]] = 0
    segmented_image[r['masks'][:, :, finger2_index]] = 0
    segmented_image[r['masks'][:, :, finger3_index]] = 0
    segmented_image[r['masks'][:, :, finger4_index]] = 0
else:
    raise Exception("Wrong number of regions ({}) detected!".format(len(r['rois'][:, 0])))

cv2.imwrite('segmented_{}'.format(opt.image_name), segmented_image)


print("Area of finger1: {}".format(area1))
if (n_regions == 5):
    print("Area of finger2: {}".format(area2))
    print("Area of finger3: {}".format(area3))
    print("Area of finger4: {}".format(area4))
