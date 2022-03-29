# PPD: A Scalable and Efficient Parallel Primal-Dual Coordinate Descent Algorithm

The offical code of PPD: A Scalable and Efficient Parallel Primal-Dual Coordinate Descent Algorithm, Hejun Wu, Xinchuan Huang, Qiong Luo, TKDE 2020.

Method PPD is described in the paper: A Scalable and Efficient Parallel Dual Coordinate Descent Algorithm.

The sparse dataset kddb.t, news20.binary, rcv1.binary in our demo are from [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

## Usage

Our code are implemented in C++ including fit_all.cpp, dual_svm.h, dual_svm.cpp, sdca_utils.h, sdca_utils.cpp, libsvm_data.h.

To run the demo on Linux, first run Makefile in the terminal.(Note that the compiler should support c++11 and Boost C++ Libraries and OpenMp are available on your system)

If compiled successfully, you can run the algorithms implemented in C++ through fit_all.cpp, here is a piece of sample instruction:

```
>>>./result --help
>>>Allowed options:
   --help                             produce help message
   --algo arg (=serial)               set algorithm
   --train arg                        set training set
   --loss arg (=L1_svm)               set loss_type(L1_svm(default),L2_svm)
   --C arg (=0.001)                   set C(default 0.001)
   --tol arg (=1e-10)                 set tolerance(default 1e-10)
   --n_epoch arg (=120)               set maximum iter times(default 120)
   --verbose arg (=1)                 set verbose(default true)
   --n_thread arg (=4)                set number of threads(default 4)
   --batch_size arg (=8)              set number of threads(default 8)
   --n_block arg (=32)                set number of blocks(default 32)
   --H arg (=10)                      set H(default 10)
   --gamma arg (=0.33300000000000002) set gamma(default 0.3333 )
   --use_best_gamma arg (=1)          set use_best_gamma(default false)
>>>./result --algo ppd --train ./kddb.t --loss L2_svm --C 0.0001 --n_thread 32 --use_best_gamma 1
>>>strat reading data of file: ../kddb.t
...
```
## Demo

To run the demo in MATLAB, first run mex_all in the MATLAB terminal to generate the mex file.(Note that the compiler should support c++11)

If mex compiled successfully, you can run Dual_gap_Different_C.m, Dual_gap_Different_Block.m to generate plot shown as below..

![Different_C](https://github.com/IMSEMZPZ/PPD/blob/master/Different_C.jpg)

![Different_C](https://github.com/IMSEMZPZ/PPD/blob/master/Different_Block.jpg)
