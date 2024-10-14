//
// Created by 崔光远 on 2024/10/14.
//

#ifndef EM_H
#define EM_H

#include <iostream>
#include <random>

#include "ClusteringBase.h"

namespace mlpa::clst {
template <typename T>
class EM final : public ClusteringBase {
private:
    T m_model;

private:
    void EStep();
    void MStep();

public:
    EM() = delete;
    EM(int, Eigen::MatrixXd, Eigen::RowVectorXi);
    void fit() override;
    Eigen::RowVectorXi get_labels() override;
};

template<typename T>
EM<T>::EM(const int c, Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : ClusteringBase(c, std::move(X), std::move(y)),
    m_model(T(m_n_clusters, m_X.rows(), m_X)) {}

template<typename GMM>
void EM<GMM>::EStep() {
    for (int i = 0; i < m_n; i++) {
        auto cur_X = m_X.col(i);
        const double sum = m_model.get_weighted_sum(cur_X);

        for (int j = 0; j < m_n_clusters; j++) {
            const double val = m_model.get_one_weighted_value(cur_X, j);
            m_z(i, j) = val / sum;
        }
    }
}

template<typename GMM>
void EM<GMM>::MStep() {
    const int d = m_X.rows();
    for (int j = 0; j < m_n_clusters; j++) {
        const double N_hat = this->m_z.col(j).sum();

        double pi_hat = N_hat / m_n;

        Eigen::VectorXd zx = Eigen::VectorXd::Zero(d);
        for (int i = 0; i < m_n; i++) {
            zx = zx + m_z(i, j) * m_X.col(i);
        }
        Eigen::VectorXd mu_hat = zx / N_hat;

        Eigen::MatrixXd zxx = Eigen::MatrixXd::Zero(d, d);
        for (int i = 0; i < m_n; i++) {
            zxx = zxx + m_z(i, j) * (m_X.col(i) - mu_hat) * (m_X.col(i) - mu_hat).transpose();
        }
        Eigen::MatrixXd cov_hat = zxx / N_hat;

        this->m_model.set_weight(pi_hat, j);
        this->m_model.set_mu(mu_hat, j);
        this->m_model.set_cov(cov_hat, j);
    }
}

template<typename T>
void EM<T>::fit() {

    EStep();
    MStep();

    for (int i = 0; i < m_n_clusters; i++) {
        m_ctr.col(i) = this->m_model.get_mu(i);
    }
}

template<typename T>
Eigen::RowVectorXi EM<T>::get_labels() {
    Eigen::RowVectorXi labels(m_n);
    for (int i = 0; i < m_n; i++) {
        int max_idx = 0;
        for (int j = 0; j < m_n_clusters; j++) {
            if (m_z(i, j) > m_z(i, max_idx)) {
                max_idx = j;
            }
        }
        labels(i) = max_idx + 1;
    }
    return labels;
}
}

#endif //EM_H