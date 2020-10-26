

#ifndef PROJECTION2
#define PROJECTION2
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include<VertexPointBAL.h>
#include<VertexCameraBAL.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/se3.h>
#include <sophus/so3.h>

inline void projection2(const VertexCameraBAL* camera,const VertexPointBAL* point3d,Eigen::Vector2d* point2d){
    Eigen::VectorXd cam=camera->estimate();
    Eigen::Vector3d p=point3d->estimate();
    Eigen::Vector3d T(cam[3],cam[4],cam[5]);
    Eigen::Vector3d rodrigues(cam[0],cam[1],cam[2]);
    Sophus::SE3 RT(Sophus::SO3::exp(rodrigues),T);
    p=RT*p;
    Eigen::Vector2d p2;
    p2=Eigen::Vector2d(-p[0]/p[2],-p[1]/p[2]);
    double r=p2[0]*p2[0]+p2[1]*p2[1];
    p2=Eigen::Vector2d(cam[6]*(1+cam[7]*r+cam[8]*r*r)*p2[0],cam[6]*(1+cam[7]*r+cam[8]*r*r)*p2[1]);
    *point2d=p2;
  // point2d=RT*p;
}
#endif
