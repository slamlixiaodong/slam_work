//
// Created by 高翔 on 2017/12/15.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
using namespace std;
using namespace cv;
string image_file = "/home/lxd/my_code/project/undistort_image/test.png";   // 请确保路径正确

int main(int argc, char **argv) {

    // 本程序需要你自己实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变，但自己实现一遍有助于理解。
    // 畸变参数
    double k1 = -1.101300397292641, k2 = 36.80466762241471, p1 = 0.07724236962284359, p2 = 0.052735118593769455;
    // 内参
    double fx = 646.1947677814908, fy = 650.7896974470311, cx = 343.39197269121667, cy = 270.2595774582981, s=2.04203937005374;

    Mat image = imread(image_file,0);   // 图像是灰度图，CV_8UC1
    int rows = image.rows, cols = image.cols;
    Mat image_undistort = Mat(rows, cols, CV_8UC1);   // 去畸变以后的图

    // 计算去畸变后图像的内容
    for (int v = 0; v < rows; v++)
        for (int u = 0; u < cols; u++) {

            double u_distorted = 0, v_distorted = 0,x=0,y=0;
            // TODO 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted) (~6 lines)
            // start your code here
            //将图像的像素坐标系通过内参矩阵转换到相机坐标系
            x=(u-cx)/fx;
            y=(v-cy)/fy;
            //在相机坐标系下进行去畸变操作 r^2=x^2+y^2
            x=x*(1+k1*pow(x*x+y*y,1)+k2*pow(x*x+y*y,2))+2*p1*x*y+p2*(pow(x*x+y*y,1)+2*x*x);
            y=y*(1+k1*pow(x*x+y*y,1)+k2*pow(x*x+y*y,2))+2*p2*x*y+p1*(pow(x*x+y*y,1)+2*y*y);
            //去畸变操作结束后，将相机坐标系重新转换到图像像素坐标系
            u_distorted=x*fx+cx;
            v_distorted=y*fy+cy;
            // end your code here

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }

    // 画图去畸变后图像
    imshow("image distorted",image);
    imshow("image undistorted", image_undistort);
    waitKey();

    return 0;
}
