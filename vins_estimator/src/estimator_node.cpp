/*ROS 节点函数，回调函数*/
#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

/*
VINSVINS的估计器初始化:第五章（V. ESTIMATOR INITIALIZATION）
基于滑动窗口的非线性优化实现紧耦合:第六章（VI. TIGHTLY-COUPLED MONOCULAR VIO）
关键帧的选择:第四章（IV. MEASUREMENT PREPROCESSING A. Vision Processing Front-end） 的部分内容
*/
Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;
// 从IMU测量值imu_msg和上一个PVQ递推得到当前PVQ
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}
/*这个函数在非线性优化时才会在process()中被调用*/
//1、从估计器中得到滑动窗口中最后一个图像帧的imu更新项[P,Q,V,ba,bg,a,g]，对imu_buf中剩余imu_msg进行PVQ递推
void update()
{
    TicToc t_predict;
    // 从估计器中得到滑动窗口中最后一个图像帧的imu更新项[P,Q,V,ba,bg,a,g]
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;
    /*2、对imu_buf中剩余的imu_msg进行PVQ递推:因为imu的频率比图像频率要高很多，
    在getMeasurements(）将图像和imu时间对齐后，imu_buf中还会存在imu数据）*/ 
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}
/*对imu和图像数据进行对齐并组合，返回的是(IMUs, img_msg)s,即图像帧所对应的所有IMU数据，并将其放入一个容器vector中。
IMU和图像帧的对应关系在新版的代码中有变化：对图像帧j，每次取完imu_buf中所有时间戳小于它的imu_msg，以及第一个时间戳
大于图像帧时间戳的imu_msg（这里还需要加上同步时间存在的延迟td）
因此在新代码中，每个大于图像帧时间戳的第一个imu_msg是被两个图像帧共用的，而产生的差别在processIMU()前进行了对应的处理。
img:    i -------- j  -  -------- k
imu:    - jjjjjjjj - j+k kkkkkkkk -
*/ 
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        //对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }
        //对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();
        //图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            //emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        /*这里把下一个imu_msg也放进去了,但没有pop
        因此当前图像帧和下一图像帧会共用这个imu_msg*/
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

/*
    4个回调函数
    这里需要注意的一点是：节点estimator，以及创建了一个process，必须考虑多线程安全问题：
    1、队列imu_buf、feature_buf、relo_buf是被多线程共享的，因而在回调函数将相应的msg放入buf或进行pop时，需要设置互斥锁m_buf，在操作前lock()，操作后unlock()。其他互斥锁同理。
    2、在feature_callback和imu_callback中还设置了条件锁，在完成将msg放入buf的操作后唤醒作用于process线程中的获取观测值数据的函数。
    3、在imu_callback中还通过lock_guard的方式构造互斥锁m_state，它能在构造时加锁，析构时解锁。
*/


/*发布最新的由IMU直接递推得到的PQV,imu回调函数，将imu_msg存入imu_buf，
递推IMU的PQV并发布"imu_propagate”
*/ 
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}
// restart回调函数，收到restart消息时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}
// relocalization回调函数，将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}



/*
    thread: visual-inertial odometry VIO主线程
    通过while (true)不断循环，主要功能包括等待并获取measurements，计算dt，然后执行以下函数：
    estimator.processIMU()进行IMU预积分
    estimator.setReloFrame()设置重定位帧
    estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化
    其中measurements的数据格式可以表示为：(IMUs, img_msg)s s表示容器（vector）
*/ 
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        /*
        1.等待上面两个接收数据完成就会被唤醒，在执行getMeasurements()
        提取measurements时互斥锁m_buf会锁住，此时无法接收数据。
        getMeasurements()的作用是对imu和图像数据进行对齐并组合，之后会具体分析
        */ 
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        // 2.对measurements中的每一个measurement (IMUs,IMG)组合进行操作
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    /*2.1 对于measurement中的每一个imu_msg，计算dt并执行processIMU()。
                    processIMU()实现了IMU的预积分，通过中值积分得到当前PQV作为优化初值*/
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    /*2.1 对于measurement中的每一个imu_msg，计算dt并执行processIMU()。
                    processIMU()实现了IMU的预积分，通过中值积分得到当前PQV作为优化初值*/
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            // 2.2在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame()
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                // 设置重定位帧
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            //2.3.建立每个特征点的(camera_id,[x,y,z,u,v,vx,vy])s的map，索引为feature_id
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                /* double x,y,z,p_u,p_v,velocity_x,velocity_y     */
                ROS_ASSERT(z == 1);///判断是否归一化了    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            // 2.4处理图像帧：视觉与IMU的初始化以及非线性优化的紧耦合
            estimator.processImage(image, img_msg->header);
            // int i=0;
            // cout<<"processImage次数"<<i++<<endl;
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            // 2.5
            /*输出：
            - 向RVIZ发布里程计信息PQV、关键点三维坐标、相机位姿、点云信息、
            IMU到相机的外参、重定位位姿等
            - 在回调函数void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
            发布最新的由IMU直接递推得到的PQV
            */
            pubOdometry(estimator, header);//"odometry"
            pubKeyPoses(estimator, header);//"key_poses"
            pubCameraPose(estimator, header);//"camera_pose"
            pubPointCloud(estimator, header);//"history_cloud"
            pubTF(estimator, header);//"extrinsic"
            pubKeyframe(estimator);//"keyframe_point"、"keyframe_pose"
            if (relo_msg != NULL)
                pubRelocalization(estimator);//"relo_relative_pose"
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        //3.更新IMU参数[P,Q,V,ba,bg,a,g]，注意线程安全
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}


//程序入口
int main(int argc, char **argv)
{
    // 初始化ROS node
    ros::init(argc, argv, "vins_estimator");
    // 创建节点句柄
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);// 读取参数，
    estimator.setParameter();//设置状态估计器参数
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");
    // 发布用于RVIZ显示的Topic，本模块具体发布的内容详见输入输出
    registerPub(n);
    /*    输入：
        1、IMU的角速度和线加速度，即订阅了IMU发布的topic：IMU_TOPIC="/imu0"
        2、图像追踪的特征点，即订阅了feature_trackers模块发布的topic：“/feature_tracker/feature"
        3、复位信号，即订阅了feature_trackers模块发布的topic：“/feature_tracker/restart"
        4、重定位的匹配点，即订阅了pose_graph模块发布的topic：“/pose_graph/match_points"
    */
    // 创建一个Subscriber，订阅名为IMU_TOPIC，注册回调函数imu_callback
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);
    // 创建VIO主线程
    std::thread measurement_process{process};
    // 循环等待回调函数
    ros::spin();

    return 0;
}
