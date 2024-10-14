//
// Created by 崔光远 on 2024/10/14.
//

#include <Eigen/LU>

#include "GMM.h"

#include <iostream>
#include <random>

namespace mlpa::clst {
GMM::GMM(const int k, const int d): m_d(d), m_k(k) {
    m_weights = Eigen::VectorXd::Ones(m_k) * (1.0 / m_k);
    m_mu = Eigen::MatrixXd::Random(m_d, m_k);
    m_cov = std::vector<Eigen::MatrixXd>(m_k);
    for (int i = 0; i < m_k; i++) {
        m_cov[i] = Eigen::MatrixXd::Identity(m_d, m_d);
    }
}

GMM::GMM(const int k, const int d, const Eigen::MatrixXd &X): m_d(d), m_k(k) {

    // Initialization is important for GMM !!!

    Eigen::MatrixXd init_mu(m_d, m_k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, static_cast<int>(X.cols()));
    for (int i = 0; i < m_k; ++i) {
        const int random_number = distr(gen);
        init_mu.col(i) = X.col(random_number);
    }

    m_weights = Eigen::VectorXd::Ones(m_k) * (1.0 / m_k);
    m_mu = std::move(init_mu);
    m_cov = std::vector<Eigen::MatrixXd>(m_k);
    for (int i = 0; i < m_k; i++) {
        m_cov[i] = Eigen::MatrixXd::Identity(m_d, m_d);
    }
}

double GMM::get_one_value(const Eigen::VectorXd &X, const int k) const {
    const Eigen::MatrixXd cov = m_cov[k];
    const Eigen::VectorXd mu = m_mu.col(k);
    const double norm_cost = pow(2 * M_PI, -m_d / 2.0) * pow(cov.determinant(), -0.5);
    Eigen::VectorXd diff = X - mu;
    const double exponent = exp((-0.5 * diff.transpose() * cov.inverse() * diff).eval().value());
    return norm_cost * exponent;
}

double GMM::get_one_weighted_value(const Eigen::VectorXd &X, const int k) const {
    return m_weights(k) * get_one_value(X, k);
}

double GMM::get_weighted_sum(const Eigen::VectorXd &X) const {
    double sum = 0.0;
    for (int i = 0; i < m_k; i++) {
        sum += get_one_weighted_value(X, i);
    }
    return sum;
}

Eigen::VectorXd GMM::get_mu(const int j) const {
    return m_mu.col(j);
}

void GMM::set_weight(const double w, const int j) {
    this->m_weights(j) = w;
}

void GMM::set_mu(const Eigen::VectorXd &m, const int j) {
    this->m_mu.col(j) = m;
}

void GMM::set_cov(Eigen::MatrixXd c, const int j) {
    this->m_cov[j] = std::move(c);
}
} // namespace mlpa::clst