#pragma once

#include "resource.h"
#include <string>
#include <set>


#define GL_DATA_PATH_LEFT_F	"D:/data/images/left_fingers/"
#define GL_RCNN_PATH		"C:/D/work/rcnn/nails/nail_images/"


const std::set<std::string>   ext_set( { ".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG" } );


int main( size_t monitorHeight, size_t monitorWidth );