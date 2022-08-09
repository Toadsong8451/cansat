#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include "./eigen-3.4.0/Eigen/Dense"
#include "./kalman.hpp"
#include<sstream>
#include<stdlib.h>
#include <fstream>
#include <string>
#include <time.h>
#include<wiringPi.h>

#define SERVO1 1
#define SERVO2 23

int threshold = 120;
int minimum_area = 50; //minimum area for detection, used for erasing noise
int mid_row = 320, mid_col = 240; //middle index of the image

int ct = 0;

cv::Vec3b low, up;
cv::Mat img;
//return current time in string
std::string NowToString()
{
    srand(time(NULL));
    int temp = rand();
    return std::to_string(temp);
}

//usually in kalman.cpp, moved for clearity
double xxx, yyy;
KalmanFilter::KalmanFilter(double dt, const Eigen::MatrixXd& A, const Eigen::MatrixXd& C, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& P)
    : A(A), C(C), Q(Q), R(R), P0(P), m(C.rows()), n(A.rows()), dt(dt), initialized(false), I(n, n), x_hat(n), x_hat_new(n)
{
    I.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
    x_hat = x0;
    P = P0;
    this->t0 = t0;
    t = t0;
    initialized = true;
}

void KalmanFilter::init() {
    x_hat.setZero();
    P = P0;
    t0 = 0;
    t = t0;
    initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd& y) {

    if (!initialized)
        throw std::runtime_error("Filter is not initialized!");

    x_hat_new = A * x_hat;
    P = A * P * A.transpose() + Q;
    K = P * C.transpose() * (C * P * C.transpose() + R).inverse();
    x_hat_new += K * (y - C * x_hat_new);
    P = (I - K * C) * P;
    x_hat = x_hat_new;

    t += dt;
}

void KalmanFilter::update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd A) {

    this->A = A;
    this->dt = dt;
    update(y);
}

double Servo1_degree = 90;
double Servo2_degree = 90;

int Servo1_load = 75;
int Servo2_load = 75;

int uncatched_frame = 0;

void move_servo(double next_x, double next_y) {
    if (++uncatched_frame > 20) {
        Servo1_load = 75; Servo2_load = 75; Servo2_degree = 90; Servo1_degree = 90;
        pwmWrite(SERVO2, Servo2_load);
        delay(20);
        pwmWrite(SERVO1, Servo1_load);
        return;
    }
    std::cout << "Move   ";
    double delta_x = next_x - mid_row;
    double delta_y = next_y - mid_col;


    double rotate_y = 9.0 * delta_y / 480.0;
    double rotate_x = 15.0 * delta_x / 640.0;

    double target_degree_y = Servo2_degree + rotate_y;
    double target_degree_x = Servo1_degree + rotate_x;
    if(target_degree_y > 90) {
	target_degree_x -= rotate_x*2;
    }

    if (target_degree_y > 0 && target_degree_y < 180) {
        pwmWrite(SERVO2, 30 + target_degree_y / 2);
        Servo2_degree += rotate_y;
    }
    if (target_degree_x > 0 && target_degree_x < 180) {
        pwmWrite(SERVO1, 30 + target_degree_x / 2);
        Servo1_degree = target_degree_x;
    }
    
    
    //std::cout << rotate_x << " " << rotate_y << std::endl;
}

int main() {
    if (wiringPiSetup() == -1) return 0;
    pinMode(SERVO1, PWM_OUTPUT);
    pinMode(SERVO2, PWM_OUTPUT);
    pwmSetMode(PWM_MODE_MS);
    pwmSetClock(384);
    pwmSetRange(1000);

    pwmWrite(SERVO2, Servo2_load);
    pwmWrite(SERVO1, Servo1_load);

    double dt = 1;
    Eigen::MatrixXd System(3, 3);
    Eigen::MatrixXd Output(1, 3);
    Eigen::MatrixXd Process(3, 3);
    Eigen::MatrixXd Measure(1, 1);
    Eigen::MatrixXd Error(3, 3);

    System << 1, dt, 0, 0, 1, dt, 0, 0, 1;
    Output << 1, 0, 0;

    Process << .05, .05, .0, .05, .05, .0, .0, .0, .0;
    Measure << 5;
    Error << .1, .1, .1, .1, 10000, 10, .1, 10, 100;

    KalmanFilter kf1(dt, System, Output, Process, Measure, Error);
    KalmanFilter kf2(dt, System, Output, Process, Measure, Error);

    Eigen::VectorXd x0(3);
    Eigen::VectorXd y0(3);

    double t = 0;

    x0 << 320, 0, 0;
    y0 << 240, 0, 0;

    kf1.init(t, x0);
    kf2.init(t, x0);

    Eigen::VectorXd x1(1);
    Eigen::VectorXd y1(1);

    low = cv::Vec3b(158, threshold, threshold);
    up = cv::Vec3b(178, 255, 255);

    cv::Mat img_hsv;

    //cv::VideoCapture cap(0, cv::CAP_DSHOW);
    cv::VideoCapture cap(-1, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        return -1;
    }

    // folder create

    std::string folderName = "picture"+NowToString();
    std::string folderCreateCommand = "mkdir " + folderName;

    system(folderCreateCommand.c_str());

    while (1) {
        std::stringstream ss;
        std::string name = "pic_";
        std::string type = ".jpg";

        ss << folderName << "/" << name << (++ct) << type;
        std::string fullPath = ss.str();
        ss.str("");

        cap.read(img);
        cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
        cv::Mat img_mask;
        cv::inRange(img_hsv, low, up, img_mask);


        cv::Mat kernel = cv::Mat(11, 11, CV_8UC1, cv::Scalar(1));
        cv::morphologyEx(img_mask, img_mask, cv::MORPH_OPEN, kernel);

        cv::Mat img_labels, stats, centroids;
        int numOfLabel = cv::connectedComponentsWithStats(img_mask, img_labels, stats, centroids, 8);

        int dist = INT32_MAX;
        int obj_index = 0;
        for (int i = 1; i < numOfLabel; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < minimum_area) continue;
            int left = stats.at<int>(i, cv::CC_STAT_LEFT);
            int top = stats.at<int>(i, cv::CC_STAT_TOP);
            int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            int mid_x = left + width / 2;
            int mid_y = top + height / 2;

            int dist_temp = (mid_x - xxx) * (mid_x - xxx) + (mid_y - yyy) * (mid_y - yyy);
            if (dist_temp < dist) {
                obj_index = i;
                dist = dist_temp;
            }
        }

        if (obj_index <= 0 || obj_index >= numOfLabel) {
            move_servo(xxx, yyy);
            ss << folderName << "/" << name << ct << "fail" << type;
            fullPath = ss.str();
            imwrite(fullPath, img);
            continue;
        }
        uncatched_frame = 0;
        int left = stats.at<int>(obj_index, cv::CC_STAT_LEFT);
        int top = stats.at<int>(obj_index, cv::CC_STAT_TOP);
        int height = stats.at<int>(obj_index, cv::CC_STAT_HEIGHT);
        int width = stats.at<int>(obj_index, cv::CC_STAT_WIDTH);
        //std::cout << left + width / 2 << ", " << top + height / 2 << std::endl;
        //std::cout << dist << std::endl << std::endl;

        t += dt;
        double xx, yy;
        xx = left + width / 2;
        yy = top + height / 2;
        x1 << xx;
        y1 << yy;
        kf1.update(x1);
        kf2.update(y1);
        xxx = kf1.state().transpose()[0] + kf1.state().transpose()[1]*0.033;
        yyy = kf2.state().transpose()[0] + kf2.state().transpose()[1]*0.033;

        std::cout << ct << std::endl;

        imwrite(fullPath, img);

        std::ofstream ofs(folderName+"/location.txt", std::ios::app);
        ofs << ct - 1 << std::endl;
        ofs << "loc : (" << xx << ", " << yy << ")" << std::endl;
        ofs << "predict : (" << xxx << ", " << yyy << ")" << std::endl;
        ofs.close();
    }
    return 0;
}
