//
// Created by 崔光远 on 2024/9/22.
//

#include "LSRegression.h"

#include <iostream>
#include <Eigen/LU>

namespace mlpa::reg {
LSRegression::LSRegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    const unsigned k=1, const unsigned t_k=1)
    : RegressionBase(std::move(X), std::move(y), k, t_k) {}

LSRegression::LSRegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    std::function<Eigen::MatrixXd(Eigen::VectorXd)> phi,
    const unsigned k=1, const unsigned t_k=1)
        : RegressionBase(std::move(X), std::move(y), std::move(phi), k, t_k) {}

void LSRegression::estimate() {
    auto Phi = transform(m_X);
    m_hat_theta = (Phi * Phi.transpose()).inverse() * Phi * m_y;
}

Eigen::VectorXd LSRegression::predict(const Eigen::MatrixXd& X_star) {
    Eigen::MatrixXd Phi = transform(X_star);
    return Phi.transpose() * m_hat_theta;
}
} // namespace mlpa::reg
