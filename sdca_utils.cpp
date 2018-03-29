/////////PangzeCheung////2018////3////21////v1/////
#include"sdca_utils.h"

double dot_dense(const vector<double>& x) 
{//w.dot(w)
	double ret = 0;
	for (int i = 0; i < x.size(); i++) {
		ret += x[i] * x[i];
	}
	return (ret);
}

double dot_dense(const vector<double>& x, const vector<double>& y)
{ // w1.dot(w2)
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
{	//sqrt(Xi.dot(Xi)),used in normalize
	double ret = 0;
	for (int i = train_Data.index[k]; i < train_Data.index[k + 1]; i++) {
		ret += train_Data.X[i] * train_Data.X[i];
	}
	return sqrt(ret);
}

double dot_sparse(const Data& train_Data, const int k, const vector<double>& w)
{//Xi.dot(w)
	double ret = 0;
	for (int i = train_Data.index[k]; i < train_Data.index[k + 1]; i++) {
		ret += train_Data.X[i] * w[train_Data.col[i]];
	}
	return (ret);
}

void normalize_data(Data& train_data) {
	//在main函数中读完数据后，先进行nomalize,然后再训练 
	int n_sample = train_data.n_sample;
	vector<double> norm(n_sample, 0.0);
	for (int i = 0; i < n_sample; ++i) {
		norm[i] = norm_2_sparse(train_data, i);
	}
	for (int j = 0; j < n_sample; ++j) {//对每一个sample进行正则化
		for (int k = train_data.index[j]; k < train_data.index[j + 1]; ++k) {//遍历一个sample j
			train_data.X[k] /= norm[j];
		}
	}
}


inline double stringToNum(const string& str)
{// only used in readLibsvm()function
	istringstream iss(str);
	double num;
	iss >> num;
	return num;
}

//read data from libsvm format,还有一个任务，扩展特征维度，这个也可以放在本函数中
//另外，index中，最后一个维度，index[n_sample+1]=X.size(),为了方便编写循环
void read_libsvm(const string filename, Data &train_Data) {
	//cout << "strat to read data of file: " << filename << endl;
	ifstream fin;
	fin.open(filename.c_str());
	int num_feature = 0;
	int num_sample = 0;

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {//每一行的开头
			num_sample++;
			train_Data.X.push_back(1.0);//每行（每个sample）的开始，同时也是上一行(上一个sample)的结束，直接在每一行的开头插入扩展的1
			train_Data.col.push_back(0);//也就是说在X[i][0]=1，即在每一个样本的开始地方插入一个1
			train_Data.index.push_back(train_Data.X.size() - 1);//标记在样本i首个元素在X中的位置为此时的X.size()-1,其实就是指向新扩展值,注意这包含了X.size()=1时候
			train_Data.Y.push_back(1);
		}
		else if (read_string == "-1" || read_string == "0") {
			num_sample++;
			train_Data.X.push_back(1.0);
			train_Data.col.push_back(0);
			train_Data.index.push_back(train_Data.X.size() - 1);
			train_Data.Y.push_back(-1);
		}
		else {//每一行中的其他字段
			int colon = read_string.find(":");
			if (colon != -1) {
				int part1 = atoi(read_string.substr(0, colon).c_str());
				double part2 = stringToNum(read_string.substr(colon + 1));
				if (part1 > num_feature) {
					num_feature = part1;
				}
				train_Data.X.push_back(part2);
				train_Data.col.push_back(part1);//part1获得的结果是从1开始计数的，注意在每一个sample(每一行的第一个位置插入了1)，所以这里就不用再减1了			
			}
		}
	}//end while(fin)
	fin.close();
	train_Data.index.push_back(train_Data.X.size());//index的第一个元素是0，最后一个元素是X.size
	train_Data.n_sample = num_sample;
	train_Data.n_feature = num_feature + 1;

	//////////////////////////////////////////////////////////////
	/*for (int i = 0; i < train_Data.n_sample; i++) {
		cout << " ";
		if (train_Data.Y[i] > 0) {
		cout << "+";
		}
		cout << train_Data.Y[i];
		for (int j = train_Data.index[i]; j < train_Data.index[i + 1]; j++) {
		cout << " " << train_Data.col[j] + 1 << ":" << train_Data.X[j];
		}
	}*/
}

int cal_omega(const Data& train_data, int n_block) {
	int block_size = ceil(train_data.n_sample / n_block);//向上取整
	std::vector<int> omega(train_data.n_feature - 1, 0.0);//omega[feature_idx-1]=;

	for (int feature_idx = 1; feature_idx<train_data.n_feature; ++feature_idx)
	{//扩展特征时候，第一个特征设为1，所以从1开始，否则最终结果为K
		for (int block_idx = 0; block_idx<n_block; ++block_idx)
		{//遍历每个block
			int idx = 0;//block内部的索引
			for (; idx<block_size; ++idx)
			{ //遍历一个block内的所有samples,判断该block是否存在非零值,如果有就退出该block的遍历
				int i = block_size*block_idx + idx;//计算出sample的全局索引，或者建立索引矩阵train_indices,train_indices[block_idx][idx]存储这这个值。
				if (i >= train_data.n_sample)//无效索引
				{
					continue;
				}
				//判断X[i][feature_idx]是否为零
				int k = train_data.index[i];
				for (; k < train_data.index[i + 1]; ++k) {
					if (feature_idx == train_data.col[k]) { break; }//一旦列索引匹配就退出查找
				}
				if (k < train_data.index[i + 1]) { break; }//说明X[i][feature_idx]!=0
			}
			if (idx < block_size)//通过第104行退出for循环，说明有非零值
			{
				++omega[feature_idx - 1];
			}
		}
	}

	//找到最大的稀疏度，平均化能收敛呢？？
	int max_omega = 0;
	for (int i = 0; i<omega.size(); ++i)
	{
		if (omega[i] > max_omega)
		{
			max_omega = omega[i];
		}
	}
	cout << "w_pi is : " << max_omega << endl;
	return max_omega;

	//平均稀疏度还是很有吸引力的。
	//int sum = 0;
	//for (int feature_idx = 0; feature_idx<omegas.size(); ++feature_idx)
	//{
	//	sum += omegas[feature_idx];
	//}
	//int ave_omega = sum / n_features;
	//std::cout << "w_pi in function :" << ave_omega<<std::endl;
	//return ave_omega;
}
double min(double a, double b) {
	return (a > b) ? b : a;
}
double max(double a, double b) {
	return (a > b) ? a : b;
}
