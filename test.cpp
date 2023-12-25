#include "PHR_alm.hpp"
#include <chrono>
#include <iostream>

using namespace std;

class Opt : public PHR_ALM {
public:
  int N_ = 5000;
  Opt() {
    // num of Equality and Inequality
    PHR_ALM::e_num_ = N_ / 2;
    PHR_ALM::ine_num_ = N_ / 2;
  }

protected:
  virtual double Cost(const Eigen::VectorXd &x) override;
  virtual Eigen::VectorXd &Hx(const Eigen::VectorXd &x,
                              Eigen::VectorXd &hx) override;
  virtual Eigen::VectorXd &Gx(const Eigen::VectorXd &x,
                              Eigen::VectorXd &gx) override;
  virtual Eigen::VectorXd &Grad(const Eigen::VectorXd &x,
                                Eigen::VectorXd &grad) override;
};
double Opt::Cost(const Eigen::VectorXd &x) {
  double cost = 0;
  for (int i = 1; i <= x.rows() / 2; i++)
    cost = cost + 100 * pow(pow(x(2 * i - 2), 2) - x(2 * i - 1), 2) +
           pow(x(2 * i - 2) - 1, 2);
  return cost;
}
Eigen::VectorXd &Opt::Hx(const Eigen::VectorXd &x, Eigen::VectorXd &hx) {
  hx = x.segment(0, e_num_).array() - 1;
  return hx;
}
Eigen::VectorXd &Opt::Gx(const Eigen::VectorXd &x, Eigen::VectorXd &gx) {
  gx = (x.segment(e_num_, ine_num_).array() - 2).square() - 4;
  return gx;
}

/*
Only you need to add grad_hx and grad_gx to grad_constrants, and calculate
the orignal grad. grad_constrants = roi_ *
(grad_hx * (Hx(x, hx) + lambda_ / roi_) + grad_gx * (Gx(x, gx) + u_ /
roi_).cwiseMax(0)); grad = grad_orignal + grad_constrants;
*/
Eigen::VectorXd &Opt::Grad(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
  grad = Eigen::VectorXd::Zero(dim_);
  Eigen::VectorXd hx, gx, grad_constrants;
  Eigen::VectorXd grad_hx, grad_gx;

  /*part1: grad_orignal*/
  for (int i = 1; i <= x.rows() / 2; i++) {
    grad(2 * i - 2) =
        grad(2 * i - 2) +
        400 * (pow(x(2 * i - 2), 2) - x(2 * i - 1)) * x(2 * i - 2) +
        2 * (x(2 * i - 2) - 1);
    grad(2 * i - 1) =
        grad(2 * i - 1) - 200 * (pow(x(2 * i - 2), 2) - x(2 * i - 1));
  }
  /*part2:
  grad_constrants = roi*{ grad_hx*(hx+lambda/roi) + grad_gx*max(gx+u/roi,0)},
  Only you need to add grad_hx and grad_gx to grad_constrants
  */
  grad_hx = Eigen::VectorXd::Zero(dim_);
  grad_gx = Eigen::VectorXd::Zero(dim_);
  grad_hx.segment(0, e_num_) = Hx(x, hx) + lambda_ / roi_;
  grad_gx.segment(e_num_, ine_num_) =
      (x.segment(e_num_, ine_num_).array() * 2 - 4) *
      (Gx(x, gx) + u_ / roi_).cwiseMax(0).array();

  /*grad = grad_orignal + grad_constrants*/
  grad_constrants = roi_ * (grad_hx + grad_gx);
  grad = grad + grad_constrants;
  return grad;
}

int main() {
  Opt opt;
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(opt.N_, 10);

  auto time_start = std::chrono::high_resolution_clock::now();
  opt.Solve(x0);
  auto time_end = std::chrono::high_resolution_clock::now();
  double time1 = std::chrono::duration_cast<std::chrono::microseconds>(
                     time_end - time_start)
                     .count();

  std::cout << x0 << std::endl;
  printf("Problem:RosenBrock\n");
  printf("Dimensions:%d\n", opt.dim_);
  printf("Iters:%d\n", opt.outer_iterations_);
  printf("use time:%.3fms\n", (double)time1 / 1000.0);
}
