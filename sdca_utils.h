//#pragma once
#ifndef SDCA_UTILS_H
#define SDCA_UTILS_H

//这个代码包含所有基本的操作，包括一些内积操作和读文件函数

#include <iostream>
#include <vector>
#include <sstream>
/////////PangzeCheung////2018////3////21////v1/////
#include <fstream>
#include <stdlib.h>
#include <math.h>//ceil
#include "libsvm_data.h"
using namespace std;
//extern int N;//注意Data 类中有这个信息
//extern int MAX_DIM;

double min(double a, double b);
double max(double a, double b);

double dot_dense(const vector<double>& x); //w.dot_dense(w)
double dot_dense(const vector<double>& x, const vector<double>& y);//w1.dot_dense(w2)
double norm_2_sparse(const Data& train_Data, int k); //sqrt(Xi.dot(Xi))
double dot_sparse(const Data& train_Data, const int k, const vector<double>& w);//Xi.dot(w)
void normalize_data(Data& train_data);

inline double stringToNum(const string& str);// only used in readLibsvm()function
void read_libsvm(const string filename, Data &train_Data);

#endif // !SDCA_UTILS_H
