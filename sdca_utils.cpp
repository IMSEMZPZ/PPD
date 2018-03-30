#include"sdca_utils.h"

double dot_dense(const vector<double>& x) 
{   // w.dot(w)
	double ret = 0;
	for (int i = 0; i < x.size(); i++) {
		ret += x[i] * x[i];
	}
	return (ret);
}

double dot_dense(const vector<double>& x, const vector<double>& y)
{   // w1.dot(w2)
	if (x.size() != y.size()) {
		std::cout << "dot operation must perform on vectors with the same size!" << std::endl;
		return 0.0;
	}
	double ret = 0;
	for (int i = 0; i < x.size(); i++) {
		ret += x[i] * y[i];
	}
	return (ret);
}

double norm_2_sparse(const Data& train_Data, int k) 
{	// sqrt(Xi.dot(Xi)),used in normalize
	double ret = 0;
	for (int i = train_Data.index[k]; i < train_Data.index[k + 1]; i++) {
		ret += train_Data.X[i] * train_Data.X[i];
	}
	return sqrt(ret);
}

double dot_sparse(const Data& train_Data, const int k, const vector<double>& w)
{   // Xi.dot(w)
	double ret = 0;
	for (int i = train_Data.index[k]; i < train_Data.index[k + 1]; i++) {
		ret += train_Data.X[i] * w[train_Data.col[i]];
	}
	return (ret);
}

void normalize_data(Data& train_data) {
	// nomalize 
	int n_sample = train_data.n_sample;
	vector<double> norm(n_sample, 0.0);
	for (int i = 0; i < n_sample; ++i) {
		norm[i] = norm_2_sparse(train_data, i);
	}
	for (int j = 0; j < n_sample; ++j) {
		for (int k = train_data.index[j]; k < train_data.index[j + 1]; ++k) {
			train_data.X[k] /= norm[j];
		}
	}
}


inline double stringToNum(const string& str) { 
	// only used in readLibsvm()function
	istringstream iss(str);
	double num;
	iss >> num;
	return num;
}

void read_libsvm(const string filename, Data &train_Data) {
	ifstream fin;
	fin.open(filename.c_str());
	int num_feature = 0;
	int num_sample = 0;

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {
			num_sample++;
			train_Data.X.push_back(1.0);
			train_Data.col.push_back(0);
			train_Data.index.push_back(train_Data.X.size() - 1);
			train_Data.Y.push_back(1);
		}
		else if (read_string == "-1" || read_string == "0") {
			num_sample++;
			train_Data.X.push_back(1.0);
			train_Data.col.push_back(0);
			train_Data.index.push_back(train_Data.X.size() - 1);
			train_Data.Y.push_back(-1);
		}
		else {
			int colon = read_string.find(":");
			if (colon != -1) {
				int part1 = atoi(read_string.substr(0, colon).c_str());
				double part2 = stringToNum(read_string.substr(colon + 1));
				if (part1 > num_feature) {
					num_feature = part1;
				}
				train_Data.X.push_back(part2);
				train_Data.col.push_back(part1);		
			}
		}
	}
	fin.close();
	train_Data.index.push_back(train_Data.X.size());
	train_Data.n_sample = num_sample;
	train_Data.n_feature = num_feature + 1;
}

double min(double a, double b) {
	return (a > b) ? b : a;
}
double max(double a, double b) {
	return (a > b) ? a : b;
}
