
#ifndef COMPUTEJACOBIAN
#define COMPUTEJACOBIAN
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o_bal_class.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/se3.h>
#include <sophus/so3.h>
#include <cmath>
inline void computeJacobian(const VertexCameraBAL* camera,const VertexPointBAL* point3d,Eigen::Matrix<double,2,9> *jacobian){
    Eigen::VectorXd cam=camera->estimate();
    Eigen::Vector3d p1=point3d->estimate();
    Eigen::Vector3d T(cam[3],cam[4],cam[5]);
    Eigen::Vector3d rodrigues(cam[0],cam[1],cam[2]);
    Sophus::SE3 RT(Sophus::SO3::exp(rodrigues),T);
    Eigen::Vector3d p2=RT*p1;
    Eigen::Vector2d point2d=Eigen::Vector2d(-p2[0]/p2[2],-p2[1]/p2[2]);
    double r=point2d[0]*point2d[0]+point2d[1]*point2d[1];

    Eigen::Matrix<double, 3, 6> J1;
   // Eigen::Matrix<double,2,3> ep;
  //  ep<<cam[6]/p2[2],0,(-cam[6]*p2[0]/(p2[2]*p2[2])),0,cam[6]/p2[2],(-cam[6]*p2[1]/(p2[2]*p2[2]));
   Eigen:: Matrix<double,3,6> pp;
   Eigen::Matrix<double,3,3> II=Eigen::Matrix<double,3,3>::Identity();
    pp<<Sophus::SO3::hat(-p2),II;
    J1=pp;

    Eigen::Matrix<double, 2, 3> J2;
    J2<<-1/p2[2],0,p2[0]/(p2[2]*p2[2]),0,-1/p2[2],p2[1]/(p2[2]*p2[2]);

    Eigen:: Matrix<double, 2, 2> J3;
    J3<<cam[6]*(1+cam[7]*r+cam[8]*r*r)+cam[6]*point2d[0]*(2*cam[7]*point2d[0]+cam[8]*(4*pow(point2d[0],3)+4*point2d[0]*point2d[1]*point2d[1])),
            cam[6]*point2d[0]*(2*cam[7]*point2d[1]+cam[8]*(4*point2d[0]*point2d[0]*point2d[1]+4*pow(point2d[1],3))),
            cam[6]*point2d[1]*(2*cam[7]*point2d[0]+cam[8]*(4*point2d[1]*point2d[1]*point2d[0]+4*pow(point2d[0],3))),
            cam[6]*(1+cam[7]*r+cam[8]*r*r)+cam[6]*point2d[1]*(2*cam[7]*point2d[1]+cam[8]*(4*pow(point2d[1],3)+4*point2d[1]*point2d[0]*point2d[0]));
    Eigen::Matrix<double ,2,6>JJ1;

    JJ1=J3*J2*J1;
    //cout<<JJ1<<endl;
    Eigen:: Matrix<double, 2, 3> JJ2;
    JJ2<<(1+cam[7]*r+cam[8]*r*r)*point2d[0],cam[6]*r*point2d[0],cam[6]*r*r*point2d[0],
            (1+cam[7]*r+cam[8]*r*r)*point2d[1],cam[6]*r*point2d[1],cam[6]*r*r*point2d[1];

    JJ1[0][1]= JJ1[0][1]/2;
    JJ1[1][1]= JJ1[1][1]/2;
    *jacobian<<JJ1,JJ2;


}
#endif
//jacobian[0,8]=cam[6]*r*r*point2d[0];
//jacobian[0,7]=cam[6]*r*point2d[0];
//jacobian[0,6]=(1+cam[7]*r+cam[8]*r*r)*point2d[0];
//jacobian[1,8]=cam[6]*r*r*point2d[1];
//jacobian[1,7]=cam[6]*r*point2d[1];
//jacobian[1,6]=(1+cam[7]*r+cam[8]*r*r)*point2d[1];
