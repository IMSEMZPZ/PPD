/////////PangzeCheung////2018////3////21////v2/////
////该文件在3_21被鹏泽的相关文件全覆盖，参见文件夹C:\Users\513\Desktop\3_27_SC\实验\鹏泽\ZPZ_2018_3_21_v2

#include<iostream>
#include"sdca_utils.h"
#include"dual_svm.h"
#include<chrono>
#include<ctime>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;
#define toDouble boost::lexical_cast<double>
#define toInt boost::lexical_cast<int>

int main(int argc, char *argv[]) {
	//建立所有参数
	string loss, algo, train_file;
	double C;
	double tol;
	int n_epoch;
	bool verbose;
	int n_thread;
	int batch_size;
	int n_block;
	int H;
	double gamma;
	bool use_best_gamma;

	//从命令行匹配参数匹配参数
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		//这里的参数应该与默认实参数一致
		("algo", po::value<std::string>(&algo)->default_value("serial"), "set algorithm")
		("train", po::value<std::string>(&train_file), "set training set")
		("loss", po::value<std::string>(&loss)->default_value("L1_svm"), "set loss_type(L1_svm(default),L2_svm)")
		("C", po::value<double>(&C)->default_value(0.001), "set C(default 0.001)")
		("tol", po::value<double>(&tol)->default_value(1e-10), "set tolerance(default 1e-10)")
		("n_epoch", po::value<int>(&n_epoch)->default_value(120), "set maximum iter times(default 120)")
		("verbose", po::value<bool>(&verbose)->default_value(true), "set verbose(default true)")
		("n_thread", po::value<int>(&n_thread)->default_value(4), "set number of threads(default 4)")
		("batch_size", po::value<int>(&batch_size)->default_value(8), "set number of threads(default 8)")
		("n_block", po::value<int>(&n_block)->default_value(32), "set number of blocks(default 32)")
		("H", po::value<int>(&H)->default_value(10), "set H(default 10)")
		("gamma", po::value<double>(&gamma)->default_value(0.333), "set gamma(default 0.3333 )")
		("use_best_gamma", po::value<bool>(&use_best_gamma)->default_value(true), "set use_best_gamma(default false)")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}

	//读入数据
	std::cout << "strat reading data of file: " << train_file << std::endl;
	Data train_data;
	train_data.train_file = train_file;
	read_libsvm(train_file, train_data);

	//输出数据信息
	std::cout << "n_sample: " << train_data.n_sample << "  n_feture+1 : " << train_data.n_feature << std::endl;

	double num_non_zero = (double)train_data.X.size();
	double size_matrix = (double)train_data.n_feature*(double)train_data.n_sample;
	std::cout << " number of non-zero: " << num_non_zero << " matrix size: " << size_matrix << " density: " << num_non_zero / size_matrix << std::endl;

	cout << "strat normalize data of " << train_file << "..." << endl;
	normalize_data(train_data);

	std::cout << " strat to solve optimal problem..." << std::endl;
	//根据algo参数来创建类对象以及使用哪个fit函数
	if (algo == "serial") {//string直接和字面值比较？
		dual_svm clf1(loss, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
		clf1.fit_serial(train_data);
	}
	else if (algo == "cocoa") {
		if (gamma == 0.333) {//gamma中的0.333只是作为一个哨兵，如果po模块没有设置gamma的值，那就默认设置为设为1/n_block
			gamma = 1.0 / (double)n_block;
		}
		dual_svm clf4(loss, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
		clf4.fit_cocoa(train_data);
	}
	else if (algo == "psdca") {
		if (gamma == 0.333) {//gamma中的0.333只是作为一个哨兵，如果po模块没有设置gamma的值，那就就默认设为1
			gamma = 1.0;
		}
		dual_svm clf5(loss, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
		clf5.fit_parallel_SDCA(train_data);
	}
	else {
		std::cout << "oohs, unavailable algorithms!" << std::endl;
	}

	return 0;
}
