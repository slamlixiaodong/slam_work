//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.h"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "/home/lixiaodong/project/Bundle Adjustment/p3d.txt";
string p2d_file = "/home/lixiaodong/project/Bundle Adjustment/p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    //将3d坐标读取到p3d
    ifstream fin_p3d(p3d_file);
    double data_3d[3] = {0};
    for(int i = 0;i<76;i++)
    {
        for ( auto& d:data_3d )
            fin_p3d>>d;
        Eigen::Vector3d q;
        q<<data_3d[0],data_3d[1],data_3d[2];
        p3d.push_back( q );

    }
    //将2d坐标读取到p2d
    ifstream fin_p2d(p2d_file);
    double data_2d[2] = {0};
    for(int i = 0;i<76;i++)
    {
        for ( auto& d:data_2d )
            fin_p2d>>d;
        Eigen::Vector2d q;
        q<< data_2d[0],data_2d[1];
        p2d.push_back( q );

    }
    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE
            Eigen::Vector3d A=K*(T_esti.rotation_matrix()*p3d[i].matrix()+T_esti.translation());//计算Kexp（p^）P_i
            //将坐标的Z归一化
            A[0]=A[0]/A[2];
            A[1]=A[1]/A[2];
            A[2]=A[2]/A[2];
            //赋值残差向量e
            double x=p2d[i][0]-A[0];
            double y=p2d[i][1]-A[1];
            Eigen::Vector2d e(x,y);
            //计算代价
            cost+=e.transpose()*e;

	        // END YOUR CODE HERE

	    // compute jacobian
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE
            //计算雅阁比矩阵
            double Z=p3d[i][2]*p3d[i][2];
            double X=p3d[i][0]*p3d[i][0];
            double Y=p3d[i][1]*p3d[i][1];
            double XY=p3d[i][0]*p3d[i][1];
            J<< 0,0,0,0,0,0,
                    0,0,0,0,0,0;
            J<<fx/p3d[i][2],0,-fx*p3d[i][0]/Z,-fx*XY/Z,fx+fx*X/Z,-fx*p3d[i][1]/p3d[i][2],
                    0,fy/p3d[i][2],-fy*p3d[i][1]/Z,-fy-fy*Y/Z,fy*XY/Z,fy*p3d[i][0]/p3d[i][2];
            J=-J;
	    // END YOUR CODE HERE

            H += J.transpose() * J;//H矩阵6×6
            b += -J.transpose() * e;//b矩阵
        }

	// solve dx 
        Vector6d dx;

        // START YOUR CODE HERE
        dx=H.ldlt().solve(b);
        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE
        //增量更新
        Sophus::SE3 SE3_updated = Sophus::SE3::exp(dx)*T_esti;
        T_esti=SE3_updated;
        // END YOUR CODE HERE

        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
