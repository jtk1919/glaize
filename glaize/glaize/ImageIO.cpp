#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <vector>

#include "ImageIO.h"

using namespace std;


string ImageIO::csvf = cfg.input_csv_file;


string ImageIO::itos(int i)
{
	stringstream ss;
	ss << i;
	return ss.str();
}


string ImageIO::ftos(double d)
{
	stringstream ss;
	ss << std::fixed;
	ss << std::setprecision(1);
	ss << d;
	return ss.str();
}

int ImageIO::last_run_file_id(string f)
{
	size_t i, j;
	j = f.find_last_of('.');
	string stub = f.substr(0, j);
	i = stub.find_last_not_of("0123456789");
	string ids = stub.substr(i);
	return config::strtoi(ids);
}


ImageIO::ImageIO()
{
	csvf = cfg.input_csv_file;
	try
	{
		_csvfs.open(csvf, ios_base::in);
	}
	catch (...)
	{
		throw invalid_argument(string("[ImageInout]  Failed to open image input file ") + csvf);
	}
}


string ImageIO::getLeftFingerF()
{
	size_t start, end;
	string lff;

	getline(_csvfs, lff, ',');
	start = lff.find_last_of("/\\");
	end = lff.find_last_of(".");
	_working_fn = lff.substr(start + 1 , end - start - 1);
	return lff;
}


string ImageIO::getLeftThumbF()
{
	string f;
	getline(_csvfs, f);
	size_t end = f.find("_image.png");
	string fn = f.substr(0, end);
	return fn;
}


string ImageIO::getFingerMask(uint8_t fid) const
{
	string f = cfg.working_dir + _working_fn + "_l" + ImageIO::itos(fid) + ".png";
	return f;
}

string ImageIO::getWriteMask(uint8_t fid) const
{
	string f = cfg.results_masks_dir + _working_fn + "_l" + ImageIO::itos(fid) + ".png";
	return f;
}

string ImageIO::getFingerSeg() const
{
	string f = cfg.working_dir + _working_fn + "_nails.png";
	return f;
}

string ImageIO::getThumbSeg() const
{
	string f = cfg.working_dir + _working_fn + "_lthumb.png";
	return f;
}

   
string ImageIO::getCsvFile() const
{
	string f = cfg.results_csv_dir + _working_fn + ".csv";
	return f;
}

string ImageIO::getTempFile() const
{
	string f = cfg.results_csv_dir + "temp.txt";
	return f;
}

string ImageIO::getComposedSet(std::pair< cv::Mat, cv::Mat> &set) const
{
	cv::Mat img, im1, im2;
	int c, r;
	double tmp;
	config cfg;

	string f = cfg.data_dir + GL_SVG_PATH + _working_fn + ".png";
	img = cv::imread(f);
	tmp = img.cols * CC_LEN_PX;
	tmp /= CC_COMPOSITION;
	c = round(tmp);
	tmp = img.rows * CC_LEN_PX;
	tmp /= CC_COMPOSITION;
	r = round(tmp);
	cv::resize(img, im1, cv::Size(c, r));

	f = string(GL_DATA_DIR) + GL_SVG_PATH + _working_fn + "_R.png";
	img = cv::imread(f);
	tmp = img.cols * CC_LEN_PX;
	tmp /= CC_COMPOSITION;
	c = round(tmp);
	tmp = img.rows * CC_LEN_PX;
	tmp /= CC_COMPOSITION;
	r = round(tmp);
	cv::resize(img, im2, cv::Size(c, r));

	set = make_pair(im1.clone(), im2.clone());
	im1.release();
	im2.release();
	img.release();

	return cfg.data_dir + GL_SVG_PATH;
}

const string ImageIO::combi_findir[] = { 
	"Left fingers combi 1", "Left fingers combi 2 and 3",
	"Left fingers combi 2 and 3", "Left fingers combi 4",
	"Left fingers combi 5 and 7", "Left fingers combi 6",
	"Left fingers combi 5 and 7", "Left fingers combi 8 and 9",
	"Left fingers combi 8 and 9", "Left fingers combi 10",
	"Left fingers combi 11", "Left fingers combi 12",
	"Left fingers combi 13 and 14", "Left fingers combi 13 and 14"
};

const string ImageIO::combi_tdir[] = {
	"Thumb combi 1 and 2", "Thumb combi 1 and 2",
	"Thumb combi 3 4 and 5", "Thumb combi 3 4 and 5",
	"Thumb combi 3 4 and 5", "Thumb combi 6 7 and 8",
	"Thumb combi 6 7 and 8", "Thumb combi 6 7 and 8",
	"Thumb combi 9 10 11 12 and 13", "Thumb combi 9 10 11 12 and 13",
	"Thumb combi 9 10 11 12 and 13", "Thumb combi 9 10 11 12 and 13",
	"Thumb combi 9 10 11 12 and 13", "Thumb combi 14"
};

string ImageIO::getNail3dFile(size_t combi, size_t fin) const
{
	string f = string(GL_DATA_DIR) + GL_NAIL3D_DIR;

	if (fin < 4)
	{
		f += combi_findir[combi] + "/f3d_" + itos(fin) + ".png";
	}
	else if (fin == 4)
	{
		f += combi_tdir[combi] + "/f3d_4.png";
	}
	else
	{
		throw invalid_argument(
			string("[ERROR ImageIO::getNail3dFile] Combi out of range: ") 
				+ itos(combi));
	}

	return f;
}

string ImageIO::geFakeNailFile(size_t combi, size_t fin) const
{
	string f = string(GL_DATA_DIR) + GL_NAIL3D_DIR;
	if (fin < 4)
	{
		f += combi_findir[combi] + "/IMG_l" + itos(fin) + ".png";
	}
	else if (fin == 4)
	{
		f += combi_tdir[combi] + "/IMG_l4.png";
	}
	else
	{
		throw invalid_argument(
			string("[ERROR ImageIO::getNail3dFile] Combi out of range: ")
			+ itos(combi));
	}

	return f;
}

int ImageIO::getFin3d(int combi, cv::Mat fin3d[], cv::Mat fake[], size_t cc[])
{
	string csv, ln;
	size_t cc_fin, cc_th;

	csv = string(GL_DATA_DIR) + GL_NAIL3D_DIR + combi_findir[combi] + "/IMG.csv";
	ifstream fs(csv, ios_base::in);
	for (size_t i = 0; i < 11; ++i)
	{
		getline(fs, ln);
	}
	fs.close();
	stringstream ss;
	ss << ln;
	ss >> cc_fin;
	cc[0] = cc_fin;

	csv = string(GL_DATA_DIR) + GL_NAIL3D_DIR + combi_tdir[combi] + "/IMG.csv";
	fs.open(csv, ios_base::in);
	for (size_t i = 0; i < 5; ++i)
	{
		getline(fs, ln);
	}
	fs.close();
	stringstream ss1;
	ss1 << ln;
	ss1 >> cc_th;
	cc[1] = cc_th;

	for (size_t i = 0; i < 5; ++i)
	{
		string f = getNail3dFile(combi, i);
		fin3d[i] = cv::imread(f);

		f = geFakeNailFile(combi, i);
		fake[i] = cv::imread(f);
	}
	return 0;
}



void ImageIO::output_csv(vector< pair< vector<float>, vector<float> > >& nail_metrics, 
							int turn_angle[],
							size_t cc[])
{
	string csvf = getCsvFile();

	cout << "outputting to: " << csvf << endl;

	ofstream ofs(csvf.c_str(), ios_base::out);
	ofs << std::fixed;
	ofs << std::setprecision(1);
	ofs << "finger ID from left small finger, left half (l) or right half (r), "
		<< "half cross section mm at 0.5mm intervals from bottom up... " << endl;
	for (size_t i = 0; i < GL_NUM_FINGERS; ++i)
	{
		ofs << i << ",l,";
		for (size_t j = 0; j < nail_metrics[i].first.size(); ++j)
		{
			ofs << nail_metrics[i].first[j] << ",";
		}
		ofs << endl;

		ofs << i << ",r,";
		for (size_t j = 0; j < nail_metrics[i].second.size(); ++j)
		{
			ofs << nail_metrics[i].second[j] << ",";
		}
		ofs << endl;
	}
	for (size_t i = 0; i < 4; ++i)
	{
		ofs << turn_angle[i] << ",";
	}
	ofs << turn_angle[4] << endl;;
	ofs << cc[0] << "," << cc[1] << endl;
}

