#include"dual_svm.h"

double dual_svm::calculate_primal(Data& train_data)const {
	double P = dot_dense(w) / 2.0;
	double ave_sum = 0.0;
	if (loss == "L1_svm") {
		double temp = 0.0;
		for (int i = 0; i < train_data.n_sample; ++i) {
			temp = 1 - train_data.Y[i] * (dot_sparse(train_data, i, w));
			ave_sum += max(0.0, temp);
		}
	}
	else if (loss == "L2_svm") {
		double temp = 0.0;
		for (int i = 0; i < train_data.n_sample; ++i) {
			temp = 1 - train_data.Y[i] * (dot_sparse(train_data, i, w));
			ave_sum += max(0.0, temp)*max(0.0, temp);
		}
	}
	P += ave_sum * C;
	return P;
}

double dual_svm::calculate_dual(Data& train_data) const {
	double D = dot_dense(w) / 2.0;
	double ave_sum = 0.0;
	if (loss == "L1_svm") {
		for (int i = 0; i < train_data.n_sample; ++i) {
			ave_sum += alpha[i];
		}
	}
	else if (loss == "L2_svm") {
		for (int i = 0; i < train_data.n_sample; ++i) {
			ave_sum += alpha[i] - alpha[i] * alpha[i] / (4.0*C);
		}
	}
	D = ave_sum - D;
	return D;
}



double dual_svm::calculate_nabla(Data& train_data, int n_sample, std::vector<double> delta_w_sum, double delta_alpha_sum, double gamma, double c)const {
	double nabla = 0;
	double partial = 0;
	for (int i = 0; i < n_sample; ++i) {
		if (train_data.Y[i] * dot_sparse(train_data, i, w) + train_data.Y[i] * dot_sparse(train_data, i, delta_w_sum) * gamma < 1.0) {
			partial += -1 * train_data.Y[i] * dot_sparse(train_data, i, delta_w_sum);
		}
	}
	nabla = c * partial + 2 * gamma *dot_dense(delta_w_sum) + 2 * dot_dense(delta_w_sum, w) - delta_alpha_sum;
	return nabla;
}

int dual_svm::calculate_max_w(const string filename, int n_sample, int n_feature, int block_size) {
	std::cout << "calculating max_w......" << endl;
	ifstream fin;
	fin.open(filename.c_str());

	int line = 0;

	std::pair<int, int> *p = new pair<int, int>[n_feature];
	for (int i = 0; i < n_feature; ++i) {
		p[i].first = 0;
		p[i].second = -1;
	}

	while (!fin.eof()) {
		string read_string;
		fin >> read_string;
		if (read_string == "+1" || read_string == "1") {
			++line;
		}
		else if (read_string == "-1" || read_string == "0") {
			++line;
		}
		else {
			int colon = read_string.find(":");
			if (colon != -1) {
				int part1 = atoi(read_string.substr(0, colon).c_str());
				int current_block_index = std::ceil(double(line) / block_size);
				if (current_block_index > p[part1].second) {
					p[part1].first++;
					p[part1].second = current_block_index;
				}
			}
		}
	}
	fin.close();
	int cal_w = 0;
	for (int i = 0; i < n_feature; ++i) {
		if (p[i].first > cal_w) cal_w = p[i].first;
	}
	delete[]p;
	return cal_w;
}

void dual_svm::fit_serial(Data& train_data) {
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;
	std::cout << "n_sample" << n_sample << "  n_feature" << n_feature << endl;
	alpha = vector<double>(n_sample, 0.0);
	w = vector<double>(n_feature, 0.0);
	
	double U, Dii;
	if (loss == "L2_svm") {
		U = 1e5;
		Dii = 0.5 / C;
	}
	else if (loss == "L1_svm") {
		U = C;
		Dii = 0.0;
	}
	else {
		std::cout << "Error! Not available loss type." << std::endl;
	}
	
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, n_sample - 1);

	for (int epoch = 0; epoch < n_epoch; ++epoch)
	{
		double dual_gap = 0.0;
		double err = 0.0;
		for (int idx = 0; idx < n_sample; ++idx)
		{
			int rand_id = distribution(generator);
			double g = dot_sparse(train_data, rand_id, w)*train_data.Y[rand_id] - 1 + Dii * alpha[rand_id];
			double pg = g;
			if (std::abs(alpha[rand_id]) < tol)
			{
				pg = min(g, 0.0);
			}
			else if (std::abs(U - alpha[rand_id]) < tol)
			{
				pg = max(g, 0.0);
			}

			if (std::abs(pg) > err)
			{
				err = std::abs(pg);
			}
			if (abs(pg) > tol)
			{
				double d = min(max(alpha[rand_id] - g / (1.0 + Dii), 0.0), U) - alpha[rand_id];
				alpha[rand_id] += d;
				for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
					w[train_data.col[k]] += d * train_data.Y[rand_id] * train_data.X[k];
				}
			}
		}


		double Primal_val = calculate_primal(train_data);
		double Dual_val = calculate_dual(train_data);
		dual_gap = Primal_val - Dual_val;

		if (verbose) {
			Primal_val_array.push_back(Primal_val);
			Dual_val_array.push_back(Dual_val);
			dual_gap_array.push_back(dual_gap);
			cout << "epoch " << ": " << epoch
				<< " error:" << err
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(10) << Primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << Dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}
		if (std::abs(dual_gap) < tol)
		{
			break;
		}
	}

}

void dual_svm::fit_cocoa(Data& train_data) {
	omp_set_num_threads(n_thread);
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;
	std::cout << "n_sample= " << n_sample << "  n_feature= " << n_feature << endl;

	double U = 0.0;
	if (loss == "L2_svm") {
		U = (double)1e6;
	}
	else if (loss == "L1_svm") {
		U = C;
	}
	else {
		std::cout << "Error! Not available loss type." << std::endl;
	}
	
	for (int i = 0; i < n_sample; ++i) { alpha.push_back(0.0); }
	for (int i = 0; i < n_feature; ++i) { w.push_back(0.0); }
	
	std::vector<double>delta_alpha(n_sample, 0.0);
	int block_size = std::ceil(n_sample / n_block);
	std::vector<std::vector<double>> delta_w(n_block, std::vector<double>(n_feature));

	double sigma = gamma * n_block;
	double dual_gap;
	std::chrono::system_clock::time_point t1, t2;
	
	for (int epoch = 0; epoch < n_epoch; ++epoch) {
		dual_gap = 0.0;
		for (int j = 0; j < delta_alpha.size(); ++j) { delta_alpha[j] = 0.0; }	
		t1 = NOW;
#pragma omp parallel for num_threads(n_thread) 
		for (int block_idx = 0; block_idx < n_block; ++block_idx) {
			for (int j = 0; j < delta_w[block_idx].size(); ++j) { delta_w[block_idx][j] = 0.0; }//delta_w.setZero();
			std::random_device rd;
			std::default_random_engine generator(rd());
			std::uniform_int_distribution<int> distribution(block_idx*block_size, (block_idx + 1)*block_size - 1);

			for (int idx = 0; idx < block_size*H; ++idx)
			{
				int rand_id = distribution(generator);
				if (block_idx == 2)
				{

					if (idx >= 1)
						idx = idx;
				}
				if (rand_id >= n_sample)
				{
					continue;
				}
				double temp_alpha = alpha[rand_id] + delta_alpha[rand_id];
				double G = 0.0;

				if (loss == "L1_svm") {
					G = dot_sparse(train_data, rand_id, w) + sigma * dot_sparse(train_data, rand_id, delta_w[block_idx]);
					G = G * train_data.Y[rand_id] - 1;
				}
				else if (loss == "L2_svm") {
					G = dot_sparse(train_data, rand_id, w) + sigma * dot_sparse(train_data, rand_id, delta_w[block_idx]);
					G = G * train_data.Y[rand_id] - 1 + temp_alpha / (2.0*C);
				}

				double PG = G;
				if (std::abs(temp_alpha) < tol)
				{
					PG = min(G, 0.0);
				}
				else if (std::abs(U - temp_alpha) < tol)
				{
					PG = max(G, 0.0);
				}

				if (std::abs(PG) > tol)
				{
					double d = 0.0;
					if (loss == "L1_svm") {
						d = min(max(temp_alpha - G / sigma, 0.0), U) - temp_alpha;
					}
					else if (loss == "L2_svm") {
						double Denominator = sigma + 1.0 / (2.0*C);
						d = min(max(temp_alpha - G / Denominator, 0.0), U) - temp_alpha;
					}
					delta_alpha[rand_id] += d;
					for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
						if (train_data.col[k] > 300)
						{
							train_data.col[k] = train_data.col[k];
						}
						if (k > 49749)
						{
							k = k;
						}
						delta_w[block_idx][train_data.col[k]] += d * train_data.Y[rand_id] * train_data.X[k];
					}
				}
			}
		}

		for (int j = 0; j < alpha.size(); ++j) {
			alpha[j] += gamma * delta_alpha[j];
		}
		
		for (int j = 0; j < w.size(); ++j) {
			double sum = 0.0;
			for (int k = 0; k < delta_w.size(); ++k) {
				sum += delta_w[k][j];
			}
			w[j] += gamma * sum;
		}

		double Primal_val = calculate_primal(train_data);
		double Dual_val = calculate_dual(train_data);
		dual_gap = Primal_val - Dual_val;

		if (verbose) {
            Primal_val_array.push_back(Primal_val);
			Dual_val_array.push_back(Dual_val);
			dual_gap_array.push_back(dual_gap);
			cout << "epoch " << ": " << epoch << " EpochTime: " << std::chrono::duration<double>(NOW - t1).count() << "s "
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(10) << Primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << Dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}

		if (std::abs(dual_gap) < tol)
		{
			break;
		}
	}
}

void dual_svm::fit_pdd(Data& train_data) {
	omp_set_num_threads(n_thread);
	int n_sample = train_data.n_sample, n_feature = train_data.n_feature;
	std::cout << "n_sample= " << n_sample << "  n_feature= " << n_feature << endl;
	double U;
	int block_size = std::ceil(double(n_sample) / n_block);
	int max_w = calculate_max_w(train_data.train_file, n_sample, n_feature, block_size);
	std::cout << "max_w = " << max_w << endl;
	int w_pi = max_w;
	double non_zero = train_data.X.size();
	double w_ave = non_zero / (double)train_data.n_feature;
	if (w_ave < max_w) {
		w_pi = w_ave + 0.4*(max_w - w_ave);
	}

	if (loss == "L2_svm") {
		U = (double)1e6;
	}
	else if (loss == "L1_svm") {
		U = C;
	}
	else {
		std::cout << "Error! Not available loss type." << std::endl;
	}

	alpha = vector<double>(n_sample, 0);
	w = vector<double>(n_feature, 0);
	std::vector<double>delta_alpha(n_sample, 0.0);

	std::vector<std::vector<double> > delta_w(n_block, std::vector<double>(n_feature));

	double dual_gap;
	std::chrono::system_clock::time_point t1, t2;
	for (int epoch = 0; epoch < n_epoch; ++epoch) {
		dual_gap = 0.0;
		for (int j = 0; j < delta_alpha.size(); ++j) { delta_alpha[j] = 0.0; }
		t1 = NOW;

#pragma omp parallel for num_threads(n_thread) 
		for (int block_idx = 0; block_idx < n_block; ++block_idx) {
			for (int j = 0; j < delta_w[block_idx].size(); ++j) { delta_w[block_idx][j] = 0.0; }
			std::random_device rd;
			std::default_random_engine generator(rd());
			std::uniform_int_distribution<int> distribution(block_idx*block_size, (block_idx + 1)*block_size - 1);

			for (int idx = 0; idx < block_size*H; ++idx)
			{
				int rand_id = distribution(generator);
				if (rand_id >= n_sample)
				{
					continue;
				}
				double temp_alpha = alpha[rand_id] + delta_alpha[rand_id];
				double G = 0.0;

				if (loss == "L1_svm")
				{
					G = dot_sparse(train_data, rand_id, w) + w_pi * dot_sparse(train_data, rand_id, delta_w[block_idx]);
					G = G * train_data.Y[rand_id] - 1;
				}
				else if (loss == "L2_svm")
				{
					G = dot_sparse(train_data, rand_id, w) + w_pi * dot_sparse(train_data, rand_id, delta_w[block_idx]);
					G = G * train_data.Y[rand_id] - 1 + temp_alpha / (2.0*C);
				}
				double PG = G;
				if (std::abs(temp_alpha) < tol)
				{
					PG = min(G, 0.0);
				}
				else if (std::abs(U - temp_alpha) < tol)
				{
					PG = max(G, 0.0);
				}

				if (std::abs(PG) > tol)
				{
					double d = 0.0;
					if (loss == "L1_svm") {
						d = min(max(temp_alpha - G / (double)w_pi, 0.0), U) - temp_alpha;
					}
					else if (loss == "L2_svm") {
						double Denominator = (double)w_pi + 1.0 / (2.0*C);
						d = min(max(temp_alpha - G / Denominator, 0.0), U) - temp_alpha;
					}
					delta_alpha[rand_id] += d;
					for (int k = train_data.index[rand_id]; k < train_data.index[rand_id + 1]; ++k) {
						delta_w[block_idx][train_data.col[k]] += d * train_data.Y[rand_id] * train_data.X[k];
					}
				
				}
			}
		} 

		double delta_alpha_sum = 0.0;
		double alpha_sum = 0.0;
	
		std::vector<double> delta_w_sum(n_feature, 0.0);
		if (true) {
			delta_alpha_sum = 0.0;
			alpha_sum = 0.0;
			for (int i = 0; i < n_sample; ++i)
			{
				delta_alpha_sum += delta_alpha[i];
				alpha_sum += alpha[i];
			}
			for (int j = 0; j < n_feature; ++j) {
				delta_w_sum[j] = 0.0;
				for (int k = 0; k < n_block; ++k) {
					delta_w_sum[j] += delta_w[k][j];
				}
			}
		
			std::cout << "delta_alpha_sum = " << delta_alpha_sum << endl;
			std::cout << "alpha_sum = " << alpha_sum << endl;
			std::cout << "delta_w_sum.dot(delta_w_sum) = " << dot_dense(delta_w_sum) << endl;
			std::cout << "delta_w_sum.dot(w) = " << dot_dense(delta_w_sum, w) << std::endl;
	
			if (true)
			{				
				if (loss == "L1_svm")
				{
					std::cout << "Initial gamma : " << gamma << endl;

					const double beta = 0.8;
					const double sigma = 0.01;
					const int max_loop = 30;
					double d = 0.01, gamma_next;
	
					for (int loop = 0; loop < max_loop; ++loop) {
						std::cout << "loop : " << loop << endl;
						double nabla = calculate_nabla(train_data, n_sample, delta_w_sum, delta_alpha_sum, gamma, C);
						d = nabla > 0 ? -d : d;
						gamma_next = gamma + pow(beta, loop)*d;
						std::cout << "gamma : " << gamma << endl;
						std::cout << "nabla : " << nabla << endl;
						std::cout << endl;
					}
					gamma = gamma_next;
				}
				else if (loss == "L2_svm") {
					double inf = 0, sup = 1;
					for (int loop = 0; loop < n_sample; loop++) {
						double temp = train_data.Y[loop] * dot_sparse(train_data, loop, delta_w_sum);
						if (temp > 0) {
							double temp_sup = (1.0 - train_data.Y[loop] * dot_sparse(train_data, loop, w)) / double(temp);
							sup = min(sup, temp_sup);
						}
						else if (temp < 0) {
							double temp_inf = (1 - train_data.Y[loop] * dot_sparse(train_data, loop, w)) / temp;
							inf = max(inf, temp_inf);
						}
					}
					cout <<"inf :"<< inf <<" sup: "<< sup << endl;
					if (std::abs(sup - inf) < tol) {
						gamma = (sup + inf) / 2;
					}
					else if (inf > sup) {
						double denominator = 0, numerator = 0;
						numerator = delta_alpha_sum - 2 * dot_dense(delta_w_sum, w)-0.5*dot_dense(alpha,delta_alpha)/C;
						denominator = 2 * dot_dense(delta_w_sum) + 0.5*dot_dense(delta_alpha) / C;
						if (denominator != 0) {
							gamma = numerator / denominator;
							cout << "inf > sup" << endl;
							cout << "gamma: " << gamma << endl;
						}
						else {
							cout << "inf > sup"<<endl;
							cout << "fenmu 0000000000!!!!!" << endl;
						}
					}
					else if (inf < sup) {
						double denominator = 0, numerator = 0, ans = 0, ans2 = 0;
						for (int loop = 0; loop < n_sample; loop++) {
							ans += (dot_sparse(train_data, loop, w) - train_data.Y[loop])*dot_sparse(train_data, loop, delta_w_sum);
							ans2 +=pow(dot_sparse(train_data, loop, delta_w_sum),2);
						}
						numerator = delta_alpha_sum - 2 * C*ans - 2*dot_dense(delta_w_sum, w) - 0.5*dot_dense(alpha, delta_alpha) / C;
						denominator = 2*C*ans2+2*dot_dense(delta_w_sum) + 0.5*dot_dense(delta_alpha) / C;
						if (denominator != 0) {
							gamma = numerator / denominator;
							cout << "inf < sup" << endl;
							cout << "gamma: " << gamma << endl;
						}
						else {
							cout << "fenmu 0000000000!!!!!" << endl;
						}
					}
				}
				std::cout << " gamma: " << gamma << std::endl;
			}
		}

		for (int j = 0; j < alpha.size(); ++j) {
			alpha[j] += gamma * delta_alpha[j];
		}
	
		for (int j = 0; j < w.size(); ++j) {
			double sum = 0.0;
			for (int k = 0; k < delta_w.size(); ++k) {
				sum += delta_w[k][j];
			}
			w[j] += gamma * sum;
		}
	
		double Primal_val = calculate_primal(train_data);
		double Dual_val = calculate_dual(train_data);
		dual_gap = Primal_val - Dual_val;
	
		if (verbose) {
            Primal_val_array.push_back(Primal_val);
			Dual_val_array.push_back(Dual_val);
			dual_gap_array.push_back(dual_gap);
			cout << "epoch " << ": " << epoch << " EpochTime: " << std::chrono::duration<double>(NOW - t1).count() << "s "
				<< " Primal_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(10) << Primal_val
				<< " Dual_val: " << setiosflags(ios::fixed) << setiosflags(ios::right) << Dual_val
				<< " dual_gap:  " << setiosflags(ios::fixed) << setiosflags(ios::right) << dual_gap << endl;
		}	
		if (std::abs(dual_gap) < tol)
		{
			break;
		}
	} 
}




