//
// Created by 崔光远 on 2024/10/8.
//

#include "ClusteringBase.h"

#include <iostream>

namespace mlpa::clst {
ClusteringBase::ClusteringBase(Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : m_n_clusters(1), m_X(std::move(X)), m_y(std::move(y)) {
    m_n = m_X.cols();
    m_ctr = Eigen::MatrixXd::Random(m_X.rows(), m_n_clusters);
    m_z = Eigen::MatrixXd::Zero(m_n, m_n_clusters);
    m_estimated_y = Eigen::RowVectorXi::Random(m_n);
}

ClusteringBase::ClusteringBase(const int c, Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : m_n_clusters(c), m_X(std::move(X)), m_y(std::move(y)) {
    m_n = m_X.cols();
    m_ctr = Eigen::MatrixXd::Random(m_X.rows(), m_n_clusters);
    m_z = Eigen::MatrixXd::Zero(m_n, m_n_clusters);
    m_estimated_y = Eigen::RowVectorXi::Random(m_n);
}

Eigen::MatrixXd ClusteringBase::get_centers() {
    return this->m_ctr;
}

Eigen::RowVectorXi ClusteringBase::get_labels() {
    Eigen::RowVectorXi labels(m_n);
    for (long i = 0; i < m_n; i++) {
        for (int j = 0; j < m_n_clusters; j++) {
            if (m_z(i, j) == 1) labels(i) = j + 1;
        }
    }
    return labels;
}
} // namespace mlpa::clst
