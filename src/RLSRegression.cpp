//
// Created by 崔光远 on 2024/9/23.
//

#include "RLSRegression.h"

#include <Eigen/LU>

namespace mlpa::reg {
RLSRegression::RLSRegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    const unsigned k=1, const unsigned t_k=1)
    : RegressionBase(std::move(X), std::move(y), k, t_k), m_lambda(0) {}

RLSRegression::RLSRegression(Eigen::MatrixXd X, Eigen::VectorXd y, const double l,
    const unsigned k=1, const unsigned t_k=1)
    : RegressionBase(std::move(X), std::move(y), k, t_k), m_lambda(l) {}

void RLSRegression::estimate() {
    const auto Phi = transform(m_X);
    const auto Phi_mul = (Phi * Phi.transpose()).eval();
    const auto I = Eigen::MatrixXd::Identity(Phi_mul.rows(), Phi_mul.cols());
    m_hat_theta = (Phi_mul + m_lambda * I).inverse() * Phi * m_y;
}

Eigen::VectorXd RLSRegression::predict(const Eigen::MatrixXd& X_star) {
    Eigen::MatrixXd Phi = transform(X_star);
    return Phi.transpose() * m_hat_theta;
}
}// namespace mlpa::reg