#pragma once

#include "resource.h"
#include <string>
#include <set>
#include "config.h"


const std::set<std::string>   ext_set( { ".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG" } );


int main( size_t monitorHeight, size_t monitorWidth );