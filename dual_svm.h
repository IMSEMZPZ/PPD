/////////PangzeCheung////2018////3////21////v1/////
#ifndef DUAL_SVM_H
#define DUAL_SVM_H

#include<string>
#include<vector>
#include<cmath>//ceil,sqrt
#include<random>
#include<iomanip>//output format
#include"libsvm_data.h"
#include"sdca_utils.h"//�����ִ�Сд��ô��
#include<omp.h>
#include <chrono>
#include <algorithm>//random_shuffle
#define NOW std::chrono::system_clock::now() 
//extern int N;//ע��Data �����������Ϣ
//extern int MAX_DIM;

class dual_svm {
public:
	dual_svm(std::string _loss = "L1_svm", double _C = 0.01, double _tol = 1e-10, int _n_epoch = 30, bool _verbose = true, int _n_thread = 32, int _batch_size = 8, int _n_block =32, int _H = 10, double _gamma = 0.333, bool _use_best_gamma = false) :
		loss(_loss), C(_C), tol(_tol), n_epoch(_n_epoch), verbose(_verbose), n_thread(_n_thread), batch_size(_batch_size), n_block(_n_block),H(_H), gamma(_gamma), use_best_gamma(_use_best_gamma) {}
	
	void initial_w(std::vector<double>&new_w);//��ʼ��w,һ������Ϊ0
	void initial_alpha(std::vector<double>&new_alpha);//��ʼ��alpha,һ����Ϊ��

	void update_w(std::vector<double>&new_w);//һ����˵�������ýӿ�ȥ�޸�private��Ա
	void update_alpha(std::vector<double>&new_alpha);
	
	double calculate_primal(Data& train_data)const;//����primal value,��ʵ���ļ���ҲҪ����const�ؼ��֡�
	double calculate_dual(Data& train_data)const;//����dual value
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
	//----------------------------------------------------��������
	string loss;//L1-svm,L2-svm
	double C;//penalty parameter
	double tol;
	int n_epoch;
	bool verbose;
	std::vector<double> w;
	std::vector<double> alpha;
	//----------------------------------------------------for parallel algorithms
	int n_thread;//�����߳��������ǣ�����mini_batch���ԣ����жȷ�Ӧ��batch_size�ϣ�����CoCoA���ԣ���Ӧ��block_size�ϣ����������n_threadֻ��֪ͨCPU,����Ҫ��ô�����ȥ��ʵ�顣
	//��Ȼ��passcodeҲ��������Ϊ��mini_batch������
	//-------------------------------------------for minibatch,passcode(can use or not)
	int batch_size; //b
	//-------------------------------------------for cocoa��our parallel SDCA the same as cocoa
	int n_block;//�ֿ���Ŀ����Ӧ�����ж�
	int H;//���������⾫��
	double gamma;
	bool use_best_gamma;
	//-------------------------------------------for other parameter
};
#endif // !DUAL_SVM_H
