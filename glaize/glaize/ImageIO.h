#pragma once

#include <fstream>
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

	string getCsvFile() const;

	static string itos(int i);
	static string ftos(double d);

	void output_csv(vector< pair< vector<float>, vector<float> > >  &nail_metrics);

private:

	static string csvf; 

	ifstream _csvfs;
	string _working_fn;

};