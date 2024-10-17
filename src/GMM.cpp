//
// Created by 崔光远 on 2024/10/14.
//

#include <Eigen/LU>

#include "GMM.h"

#include <iostream>
#include <random>

#include "utils.h"

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

    // Pick random m_k samples for initialization of mu
    Eigen::MatrixXd init_mu(m_d, m_k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, static_cast<int>(X.cols()));
    for (int i = 0; i < m_k; ++i) {
        const int random_number = distr(gen);
        init_mu.col(i) = X.col(random_number);
    }
    m_mu = std::move(init_mu);

    // 1 / m_k as init weight
    m_weights = Eigen::VectorXd::Ones(m_k) * (1.0 / m_k);

    // set init covariances as 1 (i == j)
    m_cov = std::vector<Eigen::MatrixXd>(m_k);
    for (int i = 0; i < m_k; i++) {
        m_cov[i] = Eigen::MatrixXd::Identity(m_d, m_d);
    }
}

double GMM::get_one_value(const Eigen::VectorXd &X, const int k) const {
    return gaussian(X, m_mu.col(k), m_cov[k]);
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