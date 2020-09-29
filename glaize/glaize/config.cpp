#include "config.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>


using namespace std;


config cfg;


string trim(std::string str)
{
	// remove trailing white space
	while (!str.empty() && isspace(str.back())) str.pop_back();

	// return residue after leading white space
	std::size_t pos = 0;
	while (pos < str.size() && isspace(str[pos])) ++pos;
	return str.substr(pos);
}

int config::strtoi(string s)
{
	int i;
	stringstream ss;
	ss << s;
	ss >> i;
	return i;
}


config::config()
	:_cfgFile(GLAIZE_CFG_FILE),
	data_dir(GL_DATA_DIR),
	working_dir(data_dir + GL_DATA_WORKING_SUBDIR),
	input_csv_file( working_dir + GL_IMAGE_INPUT_CSV),
	lfinger_data_dir(data_dir + GL_DATA_PATH_LEFT_F),
	lthumb_data_dir(data_dir + GL_DATA_PATH_LEFT_F),
	results_csv_dir( data_dir + GL_RESULTS_PATH),
	last_run_file_id(-1)
{
	string var, value;
	char line[1024];

	_cfgfs.open(_cfgFile.c_str(), ios_base::in);
	if (_cfgfs.fail())
	{
		cout << "[config()] Failed to open config file '" << _cfgFile.c_str()
			<< "'. Make sure the config file exists with read permissions." << endl;
		string err = string("[config()] Failed to open config file '") + _cfgFile.c_str();
		throw invalid_argument(err);
	}
	cout << "[config()] confg opened" << endl;

	while (_cfgfs.good())
	{
		_cfgfs.getline(line, 1023);

		string lnstr(trim(line));

		if ((lnstr[0] == '#') || (lnstr[0] == '[')  || lnstr.empty() )   //  # are comments
		{
			continue;
		}

		size_t i = lnstr.find('=');
		var = lnstr.substr(0, i);
		value = lnstr.substr(i + 1);
		value = trim(value);

		if (var.find("data_dir") != string::npos)
		{
			data_dir = value;
			working_dir = data_dir + GL_DATA_WORKING_SUBDIR;
			input_csv_file = working_dir + GL_IMAGE_INPUT_CSV;
			lfinger_data_dir = data_dir + GL_DATA_PATH_LEFT_F;
			lthumb_data_dir = data_dir + GL_DATA_PATH_LEFT_T;
			results_csv_dir = data_dir + GL_RESULTS_PATH;
		}
		else if (var.find("last_run_file_id") != string::npos)
		{
			last_run_file_id = strtoi(value);
		}
	}

}

