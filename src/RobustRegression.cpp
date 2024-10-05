//
// Created by 崔光远 on 2024/9/25.
//

#include <ClpSimplex.hpp>
#include <CoinModel.hpp>
#include <CoinPackedMatrix.hpp>

#include "RobustRegression.h"

namespace mlpa::reg {
RobustRegression::RobustRegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    const unsigned k=1, const unsigned t_k=1)
    : RegressionBase(std::move(X), std::move(y), k, t_k) {}

RobustRegression::RobustRegression(Eigen::MatrixXd X, Eigen::VectorXd y,
    std::function<Eigen::MatrixXd(Eigen::VectorXd)> phi,
    const unsigned k=1, const unsigned t_k=1)
        : RegressionBase(std::move(X), std::move(y), std::move(phi), k, t_k) {}

void RobustRegression::estimate() {
    const auto Phi = transform(m_X);

    const unsigned n_var = m_k + 1;
    Eigen::MatrixXd A(m_n * 2, n_var + m_n);
    A << -Phi.transpose(), -Eigen::MatrixXd::Identity(m_n, m_n),
    Phi.transpose(), -Eigen::MatrixXd::Identity(m_n, m_n);

    Eigen::VectorXd f(n_var + m_n);
    f << Eigen::VectorXd::Zero(n_var),
    Eigen::VectorXd::Ones(m_n);

    Eigen::VectorXd b(2 * m_n);
    b << -m_y,
    m_y;

    ClpSimplex model;
    const int numRows = static_cast<int>(2 * m_n);
    const int numCols = static_cast<int>(n_var + m_n);
    model.resize(numRows, numCols);

    Eigen::VectorXd lower_bounds = Eigen::VectorXd::Constant(n_var + m_n, -COIN_DBL_MAX);  // No lower bounds
    Eigen::VectorXd upper_bounds = Eigen::VectorXd::Constant(n_var + m_n, COIN_DBL_MAX);   // No upper bounds
    
    // Set the constraint matrix
    CoinPackedMatrix matrix;
    matrix.setDimensions(numRows, numCols);
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            matrix.modifyCoefficient(i, j, A(i, j));
        }
    }
    model.loadProblem(matrix, lower_bounds.data(), upper_bounds.data(),
        f.data(), nullptr, b.data());

    model.primal();

    const double* solution = model.primalColumnSolution();
    for (int i = 0; i < n_var; i++) m_hat_theta(i) = solution[i];
}

Eigen::VectorXd RobustRegression::predict(const Eigen::MatrixXd& X_star) {
    const auto Phi = transform(X_star);
    return Phi.transpose() * m_hat_theta;
}
} // namespace mlpa::reg
