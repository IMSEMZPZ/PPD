#include <iostream>
#include "sdca_utils.h"
#include "dual_svm.h"
#include <chrono>
#include <ctime>
#include <string.h>
#include "mex.h"
using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
	double C, tol, gamma;
	int n_epoch, n_thread, n_block, train_type, H;
	bool use_best_gamma, verbose;
	std::string train_file, loss;
	char* file = new char[mxGetN(prhs[0]) + 1];
	mxGetString(prhs[0], file, mxGetN(prhs[0]) + 1);
	train_file = "";
	for (int i = 0; i < mxGetN(prhs[0]); i++) {
		train_file += file[i];
	}
	char* in_loss = new char[mxGetN(prhs[1]) + 1];
	mxGetString(prhs[1], in_loss, mxGetN(prhs[1]) + 1);
	loss = "";
	for (int i = 0; i < mxGetN(prhs[1]); i++) {
		loss += in_loss[i];
	}
	char* algorithm = new char[mxGetN(prhs[2]) + 1];
	mxGetString(prhs[2], algorithm, mxGetN(prhs[2]) + 1);

	C = mxGetScalar(prhs[3]);
	tol = mxGetScalar(prhs[4]);
	n_epoch = mxGetScalar(prhs[5]);
	verbose = mxGetScalar(prhs[6]);
	n_thread = mxGetScalar(prhs[7]);
	n_block = mxGetScalar(prhs[8]);
	H = mxGetScalar(prhs[9]);
	gamma = mxGetScalar(prhs[10]);
	use_best_gamma = mxGetScalar(prhs[11]);
	 
	Data train_data;
	train_data.train_file = train_file;
	dual_svm clf(loss, C, tol, n_epoch, verbose, n_thread, n_block, H, gamma, use_best_gamma);
	cout << "reading file " << train_file << "..." << endl;
	read_libsvm(train_file, train_data);
	cout << " normalize data of " << train_file << "..." << endl;
	normalize_data(train_data);
	std::cout << "strat training..." << std::endl;

	if (strcmp(algorithm, "serial") == 0) {
		clf.fit_serial(train_data);
		printf("Serial\n");
	}
	else if (strcmp(algorithm, "cocoa") == 0) {
		clf.fit_cocoa(train_data);
		printf("CoCoA\n");
	}
	else if (strcmp(algorithm, "ppd") == 0) {
		clf.fit_pdd(train_data);
		printf("PDD\n");
	}
	else {
		cout << "Error" << endl;
	}

	plhs[0] = mxCreateDoubleMatrix(clf.Primal_val_array.size(), 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(clf.Primal_val_array.size(), 1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(clf.Primal_val_array.size(), 1, mxREAL);
	double* Primal_val = mxGetPr(plhs[0]);
	double* Dual_val = mxGetPr(plhs[1]);
	double* dual_gap = mxGetPr(plhs[2]);
	for (int i = 0; i < clf.Primal_val_array.size(); i++) {
		Primal_val[i] = clf.Primal_val_array[i];
		Dual_val[i] = clf.Dual_val_array[i];
		dual_gap[i] = clf.dual_gap_array[i];
	}
}
