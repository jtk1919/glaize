#pragma once

#include <fstream>
#include <opencv2//opencv.hpp>
#include "resource.h"
#include "config.h"

using namespace std;


class ImageIO
{


public:

	ImageIO();
	~ImageIO()
	{
		_csvfs.close();
	}

	string getLeftFingerF();
	string getLeftThumbF();

	string getWorkingFn() const
	{
		return _working_fn;
	}

	string getFingerMask(uint8_t fid) const;
	string getFingerClip(uint8_t fid) const;
	string getWriteMask(uint8_t fid) const;

	string getFingerSeg() const;
	string getThumbSeg() const;

	string getCsvFile() const;
	string getTempFile() const;
	string getNail3dFile( size_t combi, size_t fin) const;
	string geFakeNailFile(size_t combi, size_t fin) const;
	string getComposedSet(std::pair< cv::Mat, cv::Mat> &set) const;

	int getFin3d( int combi, cv::Mat fin3d[], cv::Mat fake[], size_t cc[]);

	static string itos(int i);
	static string ftos(double d);
	static int last_run_file_id(string f);

	void output_csv(vector< pair< vector<float>, vector<float> > >  &nail_metrics, 
								int turn_angle[],
								size_t cc[] );

private:

	static string csvf; 

	ifstream _csvfs;
	string _working_fn;

	static const string combi_findir[];
	static const string combi_tdir[];
};