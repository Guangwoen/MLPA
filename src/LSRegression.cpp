//
// Created by 崔光远 on 2024/9/22.
//

#include "LSRegression.h"

#include <iostream>
#include <Eigen/LU>

namespace mlpa::reg {
LSRegression::LSRegression(Eigen::RowVectorXd X, Eigen::VectorXd y, const unsigned k=1)
    : RegressionBase(std::move(X), std::move(y), k) {}

LSRegression::LSRegression(Eigen::RowVectorXd X, Eigen::VectorXd y,
    std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> phi, const unsigned k=1)
        : RegressionBase(std::move(X), std::move(y), std::move(phi), k) {}

void LSRegression::estimate() {
    const auto Phi = m_phi(m_X);
    m_hat_theta = (Phi * Phi.transpose()).inverse() * Phi * m_y;
}

Eigen::VectorXd LSRegression::predict(const Eigen::RowVectorXd& X_star) {
    return m_phi(X_star).transpose() * m_hat_theta;
}
} // namespace mlpa::reg
