//
// Created by 崔光远 on 2024/9/25.
//

#include <iostream>
#include <Eigen/LU>

#include "BayesianRegression.h"

namespace mlpa::reg {
BayesianRegression::BayesianRegression(Eigen::RowVectorXd X, Eigen::VectorXd y,
    const double alpha, const double sigma_s, const unsigned k=1)
    : RegressionBase(std::move(X), std::move(y), k), m_alpha(alpha), m_sigma_s(sigma_s) {}

BayesianRegression::BayesianRegression(Eigen::RowVectorXd X, Eigen::VectorXd y,
    std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> phi,
    const double alpha, const double sigma_s, const unsigned k=1)
    : RegressionBase(std::move(X), std::move(y), std::move(phi), k),
    m_alpha(alpha), m_sigma_s(sigma_s) {}

void BayesianRegression::estimate() {
    const auto Phi = m_phi(m_X);
    m_param_hat_Cov = (1.0 / m_alpha * Eigen::MatrixXd::Identity(m_k+1, m_k+1)
        + 1.0 / m_sigma_s * Phi * Phi.transpose()).inverse();
    m_param_hat_mean = 1.0 / m_sigma_s * m_param_hat_Cov * Phi * m_y;
}

double BayesianRegression::get_mean_squared_error(
    const Eigen::RowVectorXd &true_x,
    const Eigen::VectorXd &true_y) const {
        Eigen::VectorXd py(true_x.size());
        for (auto i = 0; i < true_x.size(); ++i) {
            py[i] = this->get_predict_distrib_func()(true_x[i]).first;
        }
        const Eigen::VectorXd diff = true_y - py;
        return diff.squaredNorm() / static_cast<double>(diff.size());
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
    BayesianRegression::predict_distrib(const Eigen::RowVectorXd &X_star) const {
    const auto phi = m_phi(X_star).transpose().eval();
    const auto m_hat_mean = phi * m_param_hat_mean;
    const auto m_hat_sigma_s = phi * m_param_hat_Cov * phi.transpose();

    return {m_hat_mean, m_hat_sigma_s};
}

std::function<std::pair<double, double>(double)> BayesianRegression::get_predict_distrib_func() const {
    return [this] (const double x_star) {
        Eigen::RowVectorXd x(1);
        x << x_star;
        const auto Phi_r = this->m_phi(x).transpose().eval();
        const auto mean_hat_star = (Phi_r * m_param_hat_mean).value();
        const auto sigma_hat_star = (Phi_r * m_param_hat_Cov * Phi_r.transpose()).value();
        return std::make_pair(mean_hat_star, sigma_hat_star);
    };
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> BayesianRegression::get_hat_param() {
    return {this->m_param_hat_mean, this->m_param_hat_Cov};
}

Eigen::VectorXd BayesianRegression::predict(const Eigen::RowVectorXd &) {
    throw std::runtime_error("BayesianRegression::predict not implemented");
}

std::function<double(double)> BayesianRegression::get_predict_func() const {
    throw std::runtime_error("BayesianRegression::get_predict_func not implemented");
}

Eigen::VectorXd BayesianRegression::get_hat_theta() {
    throw std::runtime_error("BayesianRegression::get_hat_theta not implemented");
}
} // namespace mlpa::reg
