/////////PangzeCheung////2018////3////21////v1/////
#ifndef DUAL_SVM_H
#define DUAL_SVM_H

#include<string>
#include<vector>
#include<cmath>//ceil,sqrt
#include<random>
#include<iomanip>//output format
#include"libsvm_data.h"
#include"sdca_utils.h"//不区分大小写的么？
#include<omp.h>
#include <chrono>
#include <algorithm>//random_shuffle
#define NOW std::chrono::system_clock::now() 
//extern int N;//注意Data 类中有这个信息
//extern int MAX_DIM;

class dual_svm {
public:
	dual_svm(std::string _loss = "L1_svm", double _C = 0.01, double _tol = 1e-10, int _n_epoch = 30, bool _verbose = true, int _n_thread = 32, int _batch_size = 8, int _n_block =32, int _H = 10, double _gamma = 0.333, bool _use_best_gamma = false) :
		loss(_loss), C(_C), tol(_tol), n_epoch(_n_epoch), verbose(_verbose), n_thread(_n_thread), batch_size(_batch_size), n_block(_n_block),H(_H), gamma(_gamma), use_best_gamma(_use_best_gamma) {}
	
	void initial_w(std::vector<double>&new_w);//初始化w,一般设置为0
	void initial_alpha(std::vector<double>&new_alpha);//初始化alpha,一般设为零

	void update_w(std::vector<double>&new_w);//一般来说，尽量用接口去修改private成员
	void update_alpha(std::vector<double>&new_alpha);
	
	double calculate_primal(Data& train_data)const;//计算primal value,在实现文件中也要加入const关键字。
	double calculate_dual(Data& train_data)const;//计算dual value
	double calculate_f(Data& train_data, int n_sample, std::vector<double> &delta_w_sum, double alpha_sum, double delta_alpha_sum, double gamma, double c)const; 
	double calculate_nabla(Data& train_data, int n_sample, std::vector<double> delta_w_sum, double delta_alpha_sum, double gamma, double c)const;
	int calculate_max_w(const string filename, int n_sample, int n_feature, int block_size);

	void fit_serial(Data& train_data);//serial
	void fit_mini_batch(Data& train_data);//mini_batch;
	void fit_passcode(Data& train_data);//passcode
	void fit_cocoa(Data& train_data);//cocoa
	void fit_parallel_SDCA(Data& train_data);//our algorithm

	std::vector<double> Primal_val_array;
	std::vector<double> Dual_val_array;
	std::vector<double> dual_gap_array;
private:
	//----------------------------------------------------基本参数
	string loss;//L1-svm,L2-svm
	double C;//penalty parameter
	double tol;
	int n_epoch;
	bool verbose;
	std::vector<double> w;
	std::vector<double> alpha;
	//----------------------------------------------------for parallel algorithms
	int n_thread;//设置线程数，但是，对于mini_batch而言，并行度反应在batch_size上，对于CoCoA而言，反应在block_size上，所以这里的n_thread只是通知CPU,我需要这么多核心去做实验。
	//当然，passcode也可以设置为像mini_batch那样。
	//-------------------------------------------for minibatch,passcode(can use or not)
	int batch_size; //b
	//-------------------------------------------for cocoa，our parallel SDCA the same as cocoa
	int n_block;//分块数目，反应出并行度
	int H;//子问题的求解精度
	double gamma;
	bool use_best_gamma;
	//-------------------------------------------for other parameter
};
#endif // !DUAL_SVM_H
