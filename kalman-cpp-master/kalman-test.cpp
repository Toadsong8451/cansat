#include <iostream>
#include <vector>
#include "./eigen-3.4.0/Eigen/Dense"
#include "./kalman.hpp"

//usually in kalman.cpp, moved for clearity
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

    if(!initialized)
        throw std::runtime_error("Filter is not initialized!");

    x_hat_new = A * x_hat;
    P = A*P*A.transpose() + Q;
    K = P*C.transpose()*(C*P*C.transpose() + R).inverse();
    x_hat_new += K * (y - C*x_hat_new);
    P = (I - K*C)*P;
    x_hat = x_hat_new;

    t += dt;
}

void KalmanFilter::update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd A) {

    this->A = A;
    this->dt = dt;
    update(y);
}

int main()
{
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

    x0 << 0, 0, 0;
    y0 << 0, 0, 0;

    kf1.init(t, x0);
    kf2.init(t, x0);

    Eigen::VectorXd x1(1);
    Eigen::VectorXd y1(1);

    while(1)
    {
        t += dt;
        double xx, yy;
        std::cin >> xx >> yy;
        x1 << xx;
        y1 << yy;
        kf1.update(x1);
        kf2.update(y1);
        double xxx, yyy;
        xxx=kf1.state().transpose()[0]+kf1.state().transpose()[1];
        yyy=kf2.state().transpose()[0]+kf2.state().transpose()[1];
        std::cout << xxx << yyy << std::endl;
    }

    return 0;
}
