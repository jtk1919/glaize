#pragma once

#include "resource.h"
#include <string>
#include <set>

#define GL_DATA_WORKING_DIR	"D:/data/test/"
#define GL_DATA_PATH_LEFT_F	"D:/data/images/left_fingers/"
#define GL_DATA_PATH_LEFT_T	"D:/data/images/left_thumbs/"
#define GL_RCNN_PATH		"C:/D/work/rcnn/nails/nail_images/"
#define GL_RESULTS_PATH		"D:/data/results/csv/"

#define CC_LEN_PX       856
#define CC_PX_PER_MM    10

#define GL_NUM_FINGERS		5



const std::set<std::string>   ext_set( { ".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG" } );


int main( size_t monitorHeight, size_t monitorWidth );