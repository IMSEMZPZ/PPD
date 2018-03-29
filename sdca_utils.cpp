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
	//��main�����ж������ݺ��Ƚ���nomalize,Ȼ����ѵ�� 
	int n_sample = train_data.n_sample;
	vector<double> norm(n_sample, 0.0);
	for (int i = 0; i < n_sample; ++i) {
		norm[i] = norm_2_sparse(train_data, i);
	}
	for (int j = 0; j < n_sample; ++j) {//��ÿһ��sample��������
		for (int k = train_data.index[j]; k < train_data.index[j + 1]; ++k) {//����һ��sample j
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

//read data from libsvm format,����һ��������չ����ά�ȣ����Ҳ���Է��ڱ�������
//���⣬index�У����һ��ά�ȣ�index[n_sample+1]=X.size(),Ϊ�˷����дѭ��
void read_libsvm(const string filename, Data &train_Data) {
	//cout << "strat to read data of file: " << filename << endl;
	ifstream fin;
	fin.open(filename.c_str());
	int num_feature = 0;
	int num_sample = 0;

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {//ÿһ�еĿ�ͷ
			num_sample++;
			train_Data.X.push_back(1.0);//ÿ�У�ÿ��sample���Ŀ�ʼ��ͬʱҲ����һ��(��һ��sample)�Ľ�����ֱ����ÿһ�еĿ�ͷ������չ��1
			train_Data.col.push_back(0);//Ҳ����˵��X[i][0]=1������ÿһ�������Ŀ�ʼ�ط�����һ��1
			train_Data.index.push_back(train_Data.X.size() - 1);//���������i�׸�Ԫ����X�е�λ��Ϊ��ʱ��X.size()-1,��ʵ����ָ������չֵ,ע���������X.size()=1ʱ��
			train_Data.Y.push_back(1);
		}
		else if (read_string == "-1" || read_string == "0") {
			num_sample++;
			train_Data.X.push_back(1.0);
			train_Data.col.push_back(0);
			train_Data.index.push_back(train_Data.X.size() - 1);
			train_Data.Y.push_back(-1);
		}
		else {//ÿһ���е������ֶ�
			int colon = read_string.find(":");
			if (colon != -1) {
				int part1 = atoi(read_string.substr(0, colon).c_str());
				double part2 = stringToNum(read_string.substr(colon + 1));
				if (part1 > num_feature) {
					num_feature = part1;
				}
				train_Data.X.push_back(part2);
				train_Data.col.push_back(part1);//part1��õĽ���Ǵ�1��ʼ�����ģ�ע����ÿһ��sample(ÿһ�еĵ�һ��λ�ò�����1)����������Ͳ����ټ�1��			
			}
		}
	}//end while(fin)
	fin.close();
	train_Data.index.push_back(train_Data.X.size());//index�ĵ�һ��Ԫ����0�����һ��Ԫ����X.size
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
	int block_size = ceil(train_data.n_sample / n_block);//����ȡ��
	std::vector<int> omega(train_data.n_feature - 1, 0.0);//omega[feature_idx-1]=;

	for (int feature_idx = 1; feature_idx<train_data.n_feature; ++feature_idx)
	{//��չ����ʱ�򣬵�һ��������Ϊ1�����Դ�1��ʼ���������ս��ΪK
		for (int block_idx = 0; block_idx<n_block; ++block_idx)
		{//����ÿ��block
			int idx = 0;//block�ڲ�������
			for (; idx<block_size; ++idx)
			{ //����һ��block�ڵ�����samples,�жϸ�block�Ƿ���ڷ���ֵ,����о��˳���block�ı���
				int i = block_size*block_idx + idx;//�����sample��ȫ�����������߽�����������train_indices,train_indices[block_idx][idx]�洢�����ֵ��
				if (i >= train_data.n_sample)//��Ч����
				{
					continue;
				}
				//�ж�X[i][feature_idx]�Ƿ�Ϊ��
				int k = train_data.index[i];
				for (; k < train_data.index[i + 1]; ++k) {
					if (feature_idx == train_data.col[k]) { break; }//һ��������ƥ����˳�����
				}
				if (k < train_data.index[i + 1]) { break; }//˵��X[i][feature_idx]!=0
			}
			if (idx < block_size)//ͨ����104���˳�forѭ����˵���з���ֵ
			{
				++omega[feature_idx - 1];
			}
		}
	}

	//�ҵ�����ϡ��ȣ�ƽ�����������أ���
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

	//ƽ��ϡ��Ȼ��Ǻ����������ġ�
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
