#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
int main() {
    Eigen::Matrix<double,2,3> matrix_23;
    Eigen::Matrix<double,3,4>matrix_result;
    Eigen::Matrix<double,2,4>matrix_24;
    matrix_23<<1,2,3,4,5,6;
    matrix_24=Eigen::Matrix<double,2,4>::Random();
    //cout << matrix_23<< endl;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_dynamic;
    matrix_dynamic=Eigen::Matrix<double,100,100>::Random();
    cout<<"matrix[100][100]:\n"<<matrix_dynamic<<"\n"<<endl;
    //使用cholesky分解
    MatrixXd CholeskyA(3,3);
    CholeskyA << 4,-1,2, -1,6,0, 2,0,5;cout << "The matrix A is" << endl << CholeskyA << endl;
    clock_t time_st=clock();
    LLT<MatrixXd> llt(CholeskyA);
    MatrixXd L = llt.matrixL();
    cout << "The Cholesky factor L is" << endl << L << endl;
    cout << "To check this, let us compute L * L.transpose()" << endl;
    cout<<"time use in Cholesky is "<<1000*(clock()-time_st)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout << L * L.transpose() << endl;
    cout << "This should equal the matrix A" << endl;
    //使用Qr分解
    clock_t time_stt=clock();
    matrix_result=matrix_23.colPivHouseholderQr().solve(matrix_24);
    cout<<"the result of Qr:\n"<<matrix_result<<endl;
    cout<<"\ntime use in Qr is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC <<"ms\n" << endl;
    return 0;
}