//
// Created by 崔光远 on 2024/10/11.
//

#include "KMeans.h"

#include <iostream>

namespace mlpa::clst {
KMeans::KMeans(Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : ClusteringBase(std::move(X), std::move(y)) {}

KMeans::KMeans(const int c, Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : ClusteringBase(c, std::move(X), std::move(y)) {}

void KMeans::fit() {
    cluster_assignment();
    estimate_center();
}

void KMeans::cluster_assignment() {
    for (int i = 0; i < m_n; i++) {
        int min_idx = -1;
        double min_value = MAXFLOAT;

        for (int j = 0; j < m_n_clusters; j++) {
            if (const double e = pow((m_X.col(i) - m_ctr.col(j)).squaredNorm(), 2); e < min_value) {
                min_value = e;
                min_idx = j;
            }
        }

        for (int j = 0; j < m_n_clusters; j++) {
            m_z(i, j) = j == min_idx ? 1 : 0;
        }
    }
}

void KMeans::estimate_center() {
    for (int j = 0; j < m_n_clusters; j++) {
        Eigen::VectorXd mul = Eigen::VectorXd::Zero(m_X.rows());
        double sum = 0.0;
        for (int i = 0; i < m_n; i++) {
            mul = mul + m_z(i, j) * m_X.col(i);
            sum += m_z(i, j);
        }
        if (sum != 0)
            m_ctr.col(j) = mul / sum;
    }
}
}
