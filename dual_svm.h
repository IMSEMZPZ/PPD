#ifndef DUAL_SVM_H
#define DUAL_SVM_H

#include<string>
#include<vector>
#include<cmath>
#include<random>
#include<iomanip>
#include"libsvm_data.h"
#include"sdca_utils.h"
#include<omp.h>
#include <chrono>
#include <algorithm>
#define NOW std::chrono::system_clock::now() 

class dual_svm {
public:
	dual_svm(std::string _loss = "L2_svm", double _C = 0.01, double _tol = 1e-10, int _n_epoch = 30, bool _verbose = true, int _n_thread = 32, int _n_block =32, int _H = 10, double _gamma = 0.333, bool _use_best_gamma = false) :
		loss(_loss), C(_C), tol(_tol), n_epoch(_n_epoch), verbose(_verbose), n_thread(_n_thread), n_block(_n_block),H(_H), gamma(_gamma), use_best_gamma(_use_best_gamma) {}
		
	double calculate_primal(Data& train_data)const;//calculate primal value
	double calculate_dual(Data& train_data)const;//calculate dual value
	double calculate_nabla(Data& train_data, int n_sample, std::vector<double> delta_w_sum, double delta_alpha_sum, double gamma, double c)const;
	int calculate_max_w(const string filename, int n_sample, int n_feature, int block_size);//calculate max w'

	void fit_serial(Data& train_data);//serial
	void fit_cocoa(Data& train_data);//CoCoA
	void fit_pdd(Data& train_data);//PDD

	std::vector<double> Primal_val_array;
	std::vector<double> Dual_val_array;
	std::vector<double> dual_gap_array;
private:
	//essential parameter
	string loss;//L1-svm,L2-svm
	double C;//penalty parameter
	double tol;
	int n_epoch;
	bool verbose;
	std::vector<double> w;
	std::vector<double> alpha;
	//-----for parallel algorithms
	int n_thread;//num threads
	//-----for CoCoA£¬PPD
	int n_block;// num block 
	int H;//
	double gamma;
	bool use_best_gamma;
};

#endif // !DUAL_SVM_H
