#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"
#include <sophus/se3.h>
#include <sophus/so3.h>

class VertexCameraBAL : public g2o::BaseVertex<9,Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::VectorXd::ConstMapType v ( update, VertexCameraBAL::Dimension );
        _estimate += v;
    }

};


class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

 void computeJacobian(const VertexCameraBAL* camera,const VertexPointBAL* point3d,Eigen::Matrix<double,2,9> *jacobian){
    Eigen::VectorXd cam=camera->estimate();
    Eigen::Vector3d p1=point3d->estimate();
    Eigen::Vector3d T(cam[3],cam[4],cam[5]);

    Eigen::Vector3d rodrigues(cam[0],cam[1],cam[2]);
    Sophus::SE3 RT(Sophus::SO3::exp(rodrigues),T);
    Eigen::Vector3d p2=RT*p1;
    Eigen::Vector2d point2d=Eigen::Vector2d(-p2[0]/p2[2],-p2[1]/p2[2]);
    double r=point2d[0]*point2d[0]+point2d[1]*point2d[1];

    Eigen::Matrix<double,3,6> J1;
    Eigen::Matrix<double,3,3> II=Eigen::Matrix<double,3,3>::Identity();
    J1<<Sophus::SO3::hat(-p2),II;

    Eigen::Matrix<double, 2, 3> J2;
    J2<<-1/p2[2],0,p2[0]/(p2[2]*p2[2]),0,-1/p2[2],p2[1]/(p2[2]*p2[2]);

    Eigen:: Matrix<double, 2, 2> J3;
    J3<<cam[6]*(1+cam[7]*r+cam[8]*r*r)+cam[6]*point2d[0]*(2*cam[7]*point2d[0]+cam[8]*(4*pow(point2d[0],3)+4*point2d[0]*point2d[1]*point2d[1])),
        cam[6]*point2d[0]*(2*cam[7]*point2d[1]+cam[8]*(4*point2d[0]*point2d[0]*point2d[1]+4*pow(point2d[1],3))),
        cam[6]*point2d[1]*(2*cam[7]*point2d[0]+cam[8]*(4*point2d[1]*point2d[1]*point2d[0]+4*pow(point2d[0],3))),
        cam[6]*(1+cam[7]*r+cam[8]*r*r)+cam[6]*point2d[1]*(2*cam[7]*point2d[1]+cam[8]*(4*pow(point2d[1],3)+4*point2d[1]*point2d[0]*point2d[0]));
    Eigen::Matrix<double ,2,6>JJ1;

    JJ1=J3*J2*J1;
    Eigen:: Matrix<double, 2, 3> JJ2;
    JJ2<<(1+cam[7]*r+cam[8]*r*r)*point2d[0],cam[6]*r*point2d[0],cam[6]*r*r*point2d[0],
         (1+cam[7]*r+cam[8]*r*r)*point2d[1],cam[6]*r*point2d[1],cam[6]*r*r*point2d[1];
    *jacobian<<JJ1,JJ2;
}
 void computeJacobian2(const VertexCameraBAL* camera,const VertexPointBAL* point3d,Eigen::Matrix<double,2,3> *jacobian){
    Eigen::VectorXd cam=camera->estimate();
    Eigen::Vector3d p1=point3d->estimate();
    Eigen::Vector3d T(cam[3],cam[4],cam[5]);
    Eigen::Vector3d rodrigues(cam[0],cam[1],cam[2]);
    Sophus::SE3 RT(Sophus::SO3::exp(rodrigues),T);
    Eigen::Vector3d p2=RT*p1;
    Eigen::Vector2d point2d=Eigen::Vector2d(-p2[0]/p2[2],-p2[1]/p2[2]);
    double r=point2d[0]*point2d[0]+point2d[1]*point2d[1];
    Eigen::Matrix<double, 3, 3> J1;
    J1=Sophus::SO3::exp(rodrigues).matrix();
    Eigen::Matrix<double, 2, 3> J2;
    J2<<-1/p2[2],0,p2[0]/(p2[2]*p2[2]),0,-1/p2[2],p2[1]/(p2[2]*p2[2]);
     Eigen::Matrix<double ,2,3>JJ1;
     JJ1=J2*J1;
    *jacobian<<JJ1;
}





class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        ( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

    }

    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }


    virtual void linearizeOplus() override {
        //使用数值求导
        const VertexCameraBAL *cam = static_cast<const VertexCameraBAL *> ( vertex(0));
        const VertexPointBAL *point = static_cast<const VertexPointBAL *> ( vertex(1));
        Eigen::Matrix<double, 2, 9> ja1;
        Eigen::Matrix<double, 2, 3> ja2;
        computeJacobian(cam, point, &ja1);
        _jacobianOplusXi = ja1;
        computeJacobian2(cam, point, &ja2);
        _jacobianOplusXj = ja2;
    }
};
