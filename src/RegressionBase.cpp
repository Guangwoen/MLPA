//
// Created by 崔光远 on 2024/9/21.
//

#include "RegressionBase.h"

#include <iostream>

namespace mlpa::reg {
RegressionBase::RegressionBase(Eigen::MatrixXd X, Eigen::VectorXd y,
    const unsigned k=1, const unsigned t_k=1)
    : m_k(k), m_t_k(t_k), m_X(std::move(X)), m_y(std::move(y)) {
    m_n = m_X.cols();
    m_theta = Eigen::VectorXd::Random(t_k);
    m_hat_theta = Eigen::VectorXd::Random(t_k);
    m_phi = [] (Eigen::MatrixXd in_x) {
        return in_x;
    };
}

RegressionBase::RegressionBase(Eigen::MatrixXd X, Eigen::VectorXd y,
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> phi,
    const unsigned k=1, const unsigned t_k=1)
    : m_k(k), m_t_k(t_k), m_X(std::move(X)), m_y(std::move(y)), m_phi(std::move(phi)) {
    m_n = m_X.cols();
    m_theta = Eigen::VectorXd::Random(t_k);
    m_hat_theta = Eigen::VectorXd::Random(t_k);
}

RegressionBase::RegressionBase(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd theta,
    std::function<Eigen::MatrixXd(Eigen::VectorXd)> phi, const unsigned k=1, const unsigned t_k=1)
    : m_k(k), m_t_k(t_k), m_X(std::move(X)), m_y(std::move(y)),
    m_theta(std::move(theta)), m_phi(std::move(phi)) {
    m_n = m_X.cols();
    m_hat_theta = Eigen::VectorXd(theta.size()).setRandom();
}

void RegressionBase::set_phi(std::function<Eigen::MatrixXd(Eigen::VectorXd)> p) {
    m_phi = std::move(p);
}

Eigen::VectorXd RegressionBase::get_hat_theta() {
    return this->m_hat_theta;
}

Eigen::MatrixXd RegressionBase::transform(const Eigen::MatrixXd &inX) const {
    const long n = inX.cols();
    Eigen::MatrixXd Phi(m_t_k, n);
    for (int i = 0; i < n; ++i) Phi.col(i) = m_phi(inX.col(i)).col(0);
    return Phi;
}

// input x (d dim), output y (double)
std::function<double(Eigen::VectorXd)> RegressionBase::get_predict_func() const {
    return [this](const Eigen::VectorXd &x_star) {
        return (this->m_phi(x_star).transpose() * m_hat_theta).value();
    };
}

double RegressionBase::get_mean_squared_error(
    const Eigen::MatrixXd &true_x,
    const Eigen::VectorXd &true_y) const {
    Eigen::VectorXd py(true_x.cols());
    for (auto i = 0; i < true_x.cols(); ++i) {
        py[i] = this->get_predict_func()(true_x.col(i));
    }
    const Eigen::VectorXd diff = true_y - py;
    return diff.squaredNorm() / static_cast<double>(diff.size());
}

unsigned RegressionBase::get_t_k() const {
    return this->m_t_k;
}
} // namespace mlpa::reg

