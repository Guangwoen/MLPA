//
// Created by 崔光远 on 2024/9/23.
//

#include <qpOASES.hpp>
#include <iostream>

#include "LASSORegression.h"

namespace mlpa::reg {
LASSORegression::LASSORegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    const unsigned k=1, const unsigned t_k=1)
    : RegressionBase(std::move(X), std::move(y), k, t_k), m_nWSR(100), m_lambda(0) {}

LASSORegression::LASSORegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    const double l, const unsigned k=1, const unsigned t_k=1)
    : RegressionBase(std::move(X), std::move(y), k, t_k), m_nWSR(100), m_lambda(l) {}

LASSORegression::LASSORegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    const double l, const int w=100, const unsigned k=1, const unsigned t_k=1)
    : RegressionBase(std::move(X), std::move(y), k, t_k), m_nWSR(w), m_lambda(l) {}

void LASSORegression::estimate() {
    const auto Phi = transform(m_X);

    const auto multPhi = (Phi * Phi.transpose()).eval();

    Eigen::MatrixXd H(multPhi.rows()*2, multPhi.cols()*2);
    H << multPhi, -multPhi,
    -multPhi, multPhi;

    const Eigen::VectorXd multPhiY = Phi * m_y;
    Eigen::VectorXd f_t(multPhiY.size()*2);
    f_t << multPhiY,
    -multPhiY;
    const Eigen::VectorXd f_h = m_lambda * Eigen::VectorXd::Ones(f_t.rows(), f_t.cols());
    const Eigen::MatrixXd f = f_h - f_t;

    const auto var_size = static_cast<qpOASES::int_t>(m_t_k * 2);
    Eigen::VectorXd tx(var_size);

    Eigen::VectorXd lb = Eigen::VectorXd::Zero(var_size);

    qpOASES::QProblemB problem(var_size);
    const qpOASES::returnValue status = problem.init(H.data(), f.data(), lb.data(), nullptr, m_nWSR);
    problem.getPrimalSolution(tx.data());

    if (status == qpOASES::SUCCESSFUL_RETURN) {
        problem.getPrimalSolution(tx.data());
    } else {
        std::cerr << "Failed to solve the QP problem. Status: " << status << std::endl;
    }

    m_hat_theta = tx.head(var_size / 2) - tx.tail(var_size - var_size / 2);
}

Eigen::VectorXd LASSORegression::predict(const Eigen::MatrixXd& X_star) {
    const auto Phi = transform(X_star);
    return Phi.transpose() * m_hat_theta;
}
} // namespace mlpa::reg
