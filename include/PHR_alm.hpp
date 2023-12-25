#ifndef _PHR_ALM_HPP_
#define _PHR_ALM_HPP_

#include "lbfgs.hpp"

class PHR_ALM {
public:
  // Nessary LBFGS params.
  lbfgs::lbfgs_parameter_t lbfgs_params_;

  // The maximum number of outer iterations of ALM,
  // default outer_max_iterations_ = 50.
  int outer_max_iterations_ = 50;

  // The maximum number of innter iterations of ALM,
  // default innter_max_iterations_ = 12000.
  int innter_max_iterations_ = 12000;

  // The final outer-iterations of ALM,
  int outer_iterations_ = 0;

  // Penalty factor, roi = min[(1+gamma)roi,belta],
  // default roi_ = 1.0.
  double roi_ = 1.0;

  // Determines the rate of the penalty factor,
  // default gamma_ = 1.0.
  double gamma_ = 1.0;

  // The upper limit of the penalty factor,
  // default belta_ = 1e+3.
  double belta_ = 1e+3;

  // Equality num
  size_t e_num_;

  // Equality Lagrange-multiplier
  Eigen::VectorXd lambda_;

  // Inequality num
  size_t ine_num_;

  // Inequality Lagrange-multiplier
  Eigen::VectorXd u_;

  // Original dim
  int dim_;

  /**
   * @brief   ALM Constructor
   */
  PHR_ALM();

  /**
   * @brief   Solve this optimization problem
   *
   * @param   x0 Initial value for iteration
   * @return  empty
   *
   * @note
   */
  void Solve(Eigen::VectorXd &x0);

private:
  /**
   * @brief   Calculate gradient and cost of x(Optimized variables).
   *
   * @param   func_data Ptr of data. You can use the optimal class or
   * functon,and so on.
   * @param   x Optimized variables.
   * @param   grad Gradient. The function will fill this variable.
   * @return  cost(Jx).
   */
  inline static double costFunctionCallback(void *func_data,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &grad);

protected:
  /**
   * @brief   Calculate Cost(original object function)
   *
   * @param   x Original Optimized variables
   * @return  Cost
   */
  virtual double Cost(const Eigen::VectorXd &x) = 0;
  /**
   * @brief   Calculate equality constraints
   *
   * @param   x Original Optimized variables
   * @param   hx Equality constraints, the functon will fill this variable.
   * @return  quote for hx
   */
  virtual Eigen::VectorXd &Hx(const Eigen::VectorXd &x,
                              Eigen::VectorXd &hx) = 0;

  /**
   * @brief   Calculate inequality constraints
   *
   * @param   x Original Optimized variables
   * @param   gx Inequality constraints, the functon will fill this variable.
   * @return  quote for gx
   */
  virtual Eigen::VectorXd &Gx(const Eigen::VectorXd &x,
                              Eigen::VectorXd &gx) = 0;

  /**
   * @brief   Calculate gradient(ALM object function).
   * Only you need to add grad_hx and grad_gx to grad_constrants, and calculate
   * the orignal grad. grad_constrants = roi_ *
   * (grad_hx * (Hx(x, hx) + lambda_ / roi_) + grad_gx * (Gx(x, gx) + u_ /
   * roi_).cwiseMax(0)). grad = grad_orignal + grad_constrants.
   *
   * @param   x ALM Optimized variables
   * @param   grad Gradient of x, the functon will fill this variable.
   * @return  quote for grad
   */
  virtual Eigen::VectorXd &Grad(const Eigen::VectorXd &x,
                                Eigen::VectorXd &grad) = 0;
};

PHR_ALM::PHR_ALM() {
  lbfgs_params_.mem_size = 32;
  lbfgs_params_.past = 3;
  lbfgs_params_.g_epsilon = 0.1;
  lbfgs_params_.min_step = 1.0e-32;
  lbfgs_params_.delta = 1.0e-6;
  lbfgs_params_.max_iterations = innter_max_iterations_;
}

void PHR_ALM::Solve(Eigen::VectorXd &x0) {
  /* ---------- prepare ---------- */
  auto iter_num_ = 0;
  auto flag_force_return = false;
  // auto force_stop_type_ = DONT_STOP;
  auto flag_still_occ = false;
  // init parameters
  dim_ = x0.rows();
  lambda_ = Eigen::VectorXd::Zero(e_num_);
  u_ = Eigen::VectorXd::Zero(ine_num_);
  /* ---------- optimize ---------- */
  double final_cost = 0;
  int iters = 0;
  Eigen::VectorXd hx, gx, grad;
  double episilo_cons = 1e-8, episilo_prec = 1e-8;
  double episilo_cons_t = 1e16, episilo_prec_t = 1e16;
  while (iters < outer_max_iterations_ &&
         (episilo_cons_t > episilo_cons || episilo_prec_t > episilo_prec)) {
    ++iters;
    // lbfgs solve
    int opt_flag =
        lbfgs::lbfgs_optimize(x0, final_cost, PHR_ALM::costFunctionCallback,
                              NULL, NULL, this, lbfgs_params_);
    /*KKT Conditions*/
    Hx(x0, hx);
    Gx(x0, gx);
    double kkt1 = hx.rows() == 0 ? 0 : hx.cwiseAbs().maxCoeff();
    double kkt2 =
        gx.rows() == 0 ? 0 : gx.cwiseMax(-u_ / roi_).cwiseAbs().maxCoeff();
    episilo_cons_t = std::max(kkt1, kkt2);
    episilo_prec_t = Grad(x0, grad).cwiseAbs().maxCoeff();

    /*Update Params*/
    lambda_ += roi_ * hx;              // lammda_
    u_ = (u_ + roi_ * gx).cwiseMax(0); // u_
    roi_ = std::min((1 + (++gamma_)) * roi_, belta_);
    lbfgs_params_.g_epsilon = std::min(lbfgs_params_.g_epsilon / 10.0, 1e-5) *
                              std::min(1.0, episilo_cons_t);
  }
  outer_iterations_ = iters;
}

inline double PHR_ALM::costFunctionCallback(void *func_data,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &grad) {
  Eigen::VectorXd hx, gx;
  auto alm_ptr = reinterpret_cast<PHR_ALM *>(func_data);

  // gradient
  alm_ptr->Grad(x, grad);

  //||hx+lamda/roi||^2
  double cost_hx =
      (alm_ptr->Hx(x, hx) + alm_ptr->lambda_ / alm_ptr->roi_).squaredNorm();
  //||max(gx+u/roi,0)||^2
  double cost_gx = (alm_ptr->Gx(x, gx) + alm_ptr->u_ / alm_ptr->roi_)
                       .cwiseMax(0)
                       .squaredNorm();
  // L = fx+0.5*roi*(cost_hx+cost_gx)
  return alm_ptr->Cost(x) + 0.5 * alm_ptr->roi_ * (cost_hx + cost_gx);
}

#endif