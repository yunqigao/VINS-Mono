#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
//https://blog.csdn.net/liuzheng1/article/details/90052050
/*它指的是空间特征点P1映射到frame1或frame2上对应的图像坐标、特征点的跟踪速度、
空间坐标等属性都封装到类FeaturePerFrame中，*/
class FeaturePerFrame//一个特征点的属性
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;//IMU与cam同步时间差
    }
    double cur_td;//imu-camera的不同步时的相差时间
    Vector3d point;//特征点空间坐标
    Vector2d uv;//特征点映射到该帧上的图像坐标
    Vector2d velocity;//特征点的跟踪速度
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};
/*管理一个特征点 ID（看到该特征点的所有帧）某feature_id下的所有FeaturePerFrame
*/
class FeaturePerId
{
  public:
    const int feature_id;//特征点id
    int start_frame;//第一次出现该特征点的帧号
    vector<FeaturePerFrame> feature_per_frame;//管理对应帧的属性

    int used_num;//出现的次数
    bool is_outlier;
    bool is_margin;
    double estimated_depth;//逆深度
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;该特征点的状态，是否被三角化

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)//以feature_id为索引，并保存了出现该角点的第一帧的id
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();//得到该特征点最后一次跟踪到的帧号
};
/*
上述是对一个特征点讨论的，如果是对所有的特征点进行讨论的话，
则可以构建list容器，存储每一个特征点，对应的示意图如下：
*/
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;//滑框内所有路标点
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif