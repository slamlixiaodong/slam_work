#include <iostream>
#include <Eigen/Core>
#include<Eigen/Dense>
#include <Eigen/Geometry>
using namespace std;
using namespace Eigen;
int main() {
    double a,b;//定义归一化参数
    Matrix<double,3,1> t1,t2,p1_c,p_w,p2_c;//p_w为世界坐标，p2_c为相机2下坐标
    t1<<0.7,1.1,0.2;
    t2<<-0.1,0.4,0.8;
    p1_c<<0.5,-0.1,0.2;
    Matrix<double,4,1> t_v1,t_v2;//定义四元数转化矩阵
    //初始化四元数
    Quaterniond Q1(0.55,0.3,0.2,0.2);
    Quaterniond Q2(-0.1,0.3,-0.7,0.2);
    //求解相机坐标和坐标点在相机2下的坐标
    p_w = Q1.normalized().toRotationMatrix().colPivHouseholderQr().solve(p1_c-t1);
    p2_c=Q2.normalized()*p_w+t2;
    cout<<p2_c.transpose()<<endl;
    return 0;
}