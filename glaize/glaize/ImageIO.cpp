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


string ImageIO::getCsvFile() const
{
	string temp(_working_fn);
	temp.replace( 0, 5, "");

	string f = cfg.results_csv_dir + temp + ".csv";
	return f;
}


void ImageIO::output_csv(vector< pair< vector<float>, vector<float> > >& nail_metrics)
{
	string csvf = getCsvFile();

	cout << "outputting to: " << csvf;

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
}

