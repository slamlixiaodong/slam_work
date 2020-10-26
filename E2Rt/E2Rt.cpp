//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/so3.h>
using namespace Eigen;



#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    // START YOUR CODE HERE
    JacobiSVD<Eigen::MatrixXd> svd(E, ComputeThinU | ComputeThinV );
    Matrix3d V = svd.matrixV(), U = svd.matrixU();
    auto A=svd.singularValues();
    A[0]=(A[0]+A[1])/2.0;
    A[1]=(A[0]+A[1])/2.0;
    A[2]=0.0;
    //cout<<"v=\n"<<V<<endl;
    //cout<<"U=\n"<<U<<endl;
    //cout<<"A=\n"<<A<<endl;
    // END YOUR CODE HERE

    // set t1, t2, R1, R2 
    // START YOUR CODE HERE
    Matrix3d t_wedge1;
    Matrix3d t_wedge2;
    Eigen::Matrix3d R_PI = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();

    Eigen::Matrix3d Z;
    Z<< A[0],0,0,
        0,A[1],0,
        0,0,A[2];
    t_wedge1=U*R_PI*Z*U.transpose();
    t_wedge2=-U*R_PI*Z*U.transpose();

    Matrix3d R1;
    Matrix3d R2;
    R1=U*R_PI.transpose()*V.transpose();
    R2=-U*R_PI.transpose()*V.transpose();
    // END YOUR CODE HERE

    cout << "R1 = \n" << R1 << endl;
    cout << "R2 = \n" << R2 << endl;
    cout << "t1 = " << Sophus::SO3::vee(t_wedge1).transpose() << endl;
    cout << "t2 = " << Sophus::SO3::vee(t_wedge2).transpose() << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = \n" << tR << endl;

    return 0;
}
