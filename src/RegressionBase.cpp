//
// Created by 崔光远 on 2024/9/21.
//

#include "RegressionBase.h"

#include <iostream>

namespace mlpa::reg {
RegressionBase::RegressionBase(Eigen::RowVectorXd X, Eigen::VectorXd y, const unsigned k=1)
    : m_k(k), m_X(std::move(X)), m_y(std::move(y)) {
    m_n = m_X.size();
    m_theta = Eigen::VectorXd::Random(m_k + 1);
    m_hat_theta = Eigen::VectorXd::Random(m_k + 1);
    m_phi = [] (Eigen::RowVectorXd in_x) {
        return in_x;
    };
}

RegressionBase::RegressionBase(Eigen::RowVectorXd X, Eigen::VectorXd y,
    std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> phi, const unsigned k=1)
    : m_k(k), m_X(std::move(X)), m_y(std::move(y)), m_phi(std::move(phi)) {
    m_n = m_X.size();
    m_theta = Eigen::VectorXd::Random(m_k + 1);
    m_hat_theta = Eigen::VectorXd::Random(m_k + 1);
}

RegressionBase::RegressionBase(Eigen::RowVectorXd X, Eigen::VectorXd y, Eigen::VectorXd theta,
    std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> phi, const unsigned k=1)
    : m_k(k), m_X(std::move(X)), m_y(std::move(y)),
    m_theta(std::move(theta)), m_phi(std::move(phi)) {
    m_n = m_X.size();
    m_hat_theta = Eigen::VectorXd(theta.size()).setRandom();
}

void RegressionBase::set_phi(std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> p) {
    m_phi = std::move(p);
}

Eigen::VectorXd RegressionBase::get_hat_theta() {
    return this->m_hat_theta;
}

std::function<double(double)> RegressionBase::get_predict_func() const {
    return [this](const double x_star) {
        Eigen::RowVectorXd x(1);
        x << x_star;
        return (this->m_phi(x).transpose() * m_hat_theta).value();
    };
}

double RegressionBase::get_mean_squared_error(const Eigen::RowVectorXd &true_x,
    const Eigen::VectorXd &true_y) const {
    Eigen::VectorXd py(true_x.size());
    for (auto i = 0; i < true_x.size(); ++i) {
        py[i] = this->get_predict_func()(true_x[i]);
    }
    const Eigen::VectorXd diff = true_y - py;
    return diff.squaredNorm() / static_cast<double>(diff.size());
}
} // namespace mlpa::reg

