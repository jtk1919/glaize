#pragma once

#include <fstream>
#include "resource.h"
#include "config.h"

using namespace std;


class ImageIO
{


public:

	size_t num_fingers;
	string working_dir;

	ImageIO();
	~ImageIO()
	{
		_csvfs.close();
	}

	string getLeftFingerF();
	string getLeftThumbF(bool tonly = false);

	string getWorkingFn() const
	{
		return _working_fn;
	}

	string getFingerMask(uint8_t fid, bool tonly = false) const;

	string getCsvFile(bool tonly = false) const;

	static string itos(int i);
	static string ftos(double d);
	static int last_run_file_id(string f);

	void output_csv(vector< pair< vector<float>, vector<float> > >  &nail_metrics);

private:

	static string csvf; 

	ifstream _csvfs;
	string _working_fn;
	string _subdir;

	string _fn;

};