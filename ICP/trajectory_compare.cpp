#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/so3.h>
// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;
using namespace Eigen;
typedef Matrix<double, 6, 1> Vector6d;
// path to trajectory file


// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>,vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> p_e, p_g;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> p_ei,p_gi;
    /// implement pose reading code
    string c_file = "/home/lxd/my_code/project/ICP/compare.txt";
    ifstream c_fin(c_file);
    double data[16]={0};
    for(int i = 0;i<612;i++)
    {
        for ( auto& d:data )
            c_fin>>d;
        Eigen::Quaterniond p = Eigen::Quaterniond(data[7],data[4],data[5],data[6]);
        Eigen::Quaterniond q = Eigen::Quaterniond(data[12],data[13],data[14],data[15]);
        Eigen::Vector3d t_p;
        Eigen::Vector3d t_q;
        t_p<<data[1],data[2],data[3];
        t_q<<data[9],data[10],data[11];
        Sophus::SE3 SE3_p(p,t_p);
        Sophus::SE3 SE3_q(q,t_q);
        p_ei.push_back(t_p);
        p_gi.push_back(t_q);
        p_e.push_back( SE3_p);
        p_g.push_back(SE3_q);

    }
    //未优化的图请去掉下面注释
    //DrawTrajectory(p_e,p_g);
    Sophus::SE3 T_esti;
    Eigen::Vector3d p_ez,p_gz;
    double N=p_ei.size();
    //计算质心
    for (int i = 0; i < N; i++)
    {
        p_ez+=p_ei[i];
        p_gz+=p_gi[i];
    }
    p_ez=p_ez/N;
    p_gz=p_gz/N;
    //计算去质心坐标
    for (int i = 0; i < N; i++)
    {
        p_ei[i]=p_ei[i].matrix()-p_ez;
        p_gi[i]=p_gi[i].matrix()-p_gz;
    }
    //计算W矩阵
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += p_ei[i] * p_gi[i].transpose();
    }
    //矩阵（SVD）分解求解旋转阵R
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_ = U* ( V.transpose() );
    //求解平移向量
    Eigen::Vector3d t_ =p_ez - R_ * p_gz.matrix();
    //将旋转矩阵R和t赋值T_esti
    T_esti.translation()=t_.matrix();
    T_esti.setRotationMatrix(R_);
    cout<<"T_esti=\n"<<T_esti.matrix()<<endl;
    // draw trajectory in pangolin
    //将p_g的坐标根据T_esti坐标转换到p_e
    for(int i=0;i<N;i++)
    {
        Vector3d M;
        M<<p_g[i].translation()[0],
        p_g[i].translation()[1],
        p_g[i].translation()[2];
        M=T_esti.rotation_matrix()*M.matrix()+T_esti.translation();
        p_g[i].translation()[0]=M[0];
        p_g[i].translation()[1]=M[1];
        p_g[i].translation()[2]=M[2];
    }
    //估计后的图像
    DrawTrajectory(p_e,p_g);
    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses1,vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2 ) {
    if (poses1.empty()||poses2.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses1.size() - 1; i++) {
            glColor3f(1 - (float) i / poses1.size(), 0.0f, (float) i / poses1.size());
            glBegin(GL_LINES);
            auto p1 = poses1[i], p2 = poses1[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for (size_t j = 0; j < poses2.size() - 1; j++) {
            glColor3f(1 - (float) j / poses2.size(), 0.0f, (float) j / poses2.size());
            glBegin(GL_LINES);
            auto p3 = poses2[j], p4 = poses2[j + 1];
            glVertex3d(p3.translation()[0], p3.translation()[1], p3.translation()[2]);
            glVertex3d(p4.translation()[0], p4.translation()[1], p4.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}
