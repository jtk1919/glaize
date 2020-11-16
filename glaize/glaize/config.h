#pragma once

#include <fstream>

#define GLAIZE_CFG_FILE		"C:/Apps/glaize/glaize_config.txt"

#define GL_DATA_DIR				"D:/data/"
#define GL_DATA_WORKING_SUBDIR	"test/"
#define GL_IMAGE_INPUT_CSV		"rec.csv"

#define GL_DATA_PATH_LEFT_F		"images_fake_nail_test/left_fingers/"
#define GL_DATA_PATH_LEFT_T		"images_fake_nail_test/left_thumbs/"
//#define GL_DATA_PATH_LEFT_F		"images/left_fingers/"
//#define GL_DATA_PATH_LEFT_T		"images/left_thumbs/"
#define GL_RESULTS_PATH			"results/csv/"	

#define GL_NAIL3D_DIR			"testref/"

#define CC_LEN_PX				856
#define CC_PX_PER_MM			10

#define GL_NUM_FINGERS			5



using namespace std;


class config
{
public:

	config();

	string data_dir;
	string working_dir;
	string input_csv_file;
	string lfinger_data_dir;
	string lthumb_data_dir;
	string results_csv_dir;
	string results_masks_dir;

	int last_run_file_id;

	static int strtoi(string s);

private:

	string	_cfgFile;
	ifstream _cfgfs;

};

extern config cfg;

