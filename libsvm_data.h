#ifndef  LIBSVM_DATA_H
#define  LIBSVM_DATA_H


#include <vector>
using namespace std;
class Data
{
public:
	vector<double> X; //Only store value X which is not zero
	vector<double> Y; //Only store value Y
	vector<int> index; //X of sample[i] are in range(index[i],index[i+1])
	vector<int> col; //column of each X 
	int n_sample; //num samples
	int n_feature; //num featutes 
	string train_file; // train file 
	Data() :n_sample(0), n_feature(0) {}  
};

#endif // ! LIBSVM_DATA_H
