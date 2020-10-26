#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to trajectory file


// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>,vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> pose1, pose2;

    /// implement pose reading code
    // start your code here (5~10 lines)
    string g_file = "/home/lxd/my_code/project/trajectory_compare/groundtruth.txt";
    string e_file = "/home/lxd/my_code/project/trajectory_compare/estimated.txt";
    ifstream g_fin(g_file);
    double data[8] = {0},e_i=0;
    for(int i = 0;i<612;i++)
    {
        for ( auto& d:data )
            g_fin>>d;
        Eigen::Quaterniond q = Eigen::Quaterniond(data[7],data[4],data[5],data[6]);
        Eigen::Vector3d t;
        t<<data[1],data[2],data[3];
        Sophus::SE3 SE3_qt1(q,t);
        pose1.push_back( SE3_qt1 );

    }
    ifstream e_fin(e_file);
    for(int i = 0;i<612;i++)
    {
        for ( auto& d:data )
            e_fin>>d;
        Eigen::Quaterniond q = Eigen::Quaterniond(data[7],data[4],data[5],data[6]);
        Eigen::Vector3d t;
        t<<data[1],data[2],data[3];
        Sophus::SE3 SE3_qt1(q,t);
        pose2.push_back( SE3_qt1 );

    }
    for(int i=0;i<612;i++)
    {
        Sophus::SE3 SE3_qt3=pose2[i].inverse()*pose1[i];
        Eigen::Matrix<double,6,1>v_61=SE3_qt3.log();
        e_i=v_61.norm()*v_61.norm()+e_i;
    }
    double RMSE=sqrt(e_i/612);
    cout<<"RMSE= "<<RMSE<<endl;
    // end your code here

    // draw trajectory in pangolin
    DrawTrajectory(pose1,pose2);
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
