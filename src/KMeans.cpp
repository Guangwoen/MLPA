//
// Created by 崔光远 on 2024/10/11.
//

#include "KMeans.h"

#include <iostream>
#include <limits>
#include <random>

namespace mlpa::clst {
KMeans::KMeans(Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : ClusteringBase(std::move(X), std::move(y)) {}

KMeans::KMeans(const int c, Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : ClusteringBase(c, std::move(X), std::move(y)) {}

void KMeans::fit(const int max_iter) {
    initialize_centroids(); // Initialize centroids
    bool has_converged = false;
    int iter = 0;

    while (!has_converged && iter < max_iter) {
        iter++;
        Eigen::MatrixXd old_centroids = m_ctr; // Store old centroids

        cluster_assignment();
        estimate_center();

        // Check for convergence
        has_converged = (old_centroids - m_ctr).norm() < C_TOLERANCE; // Define a tolerance value
    }
}

void KMeans::cluster_assignment() {
    for (long i = 0; i < m_n; i++) {
        long min_idx = -1;
        double min_value = std::numeric_limits<double>::max();

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
        for (long i = 0; i < m_n; i++) {
            mul = mul + m_z(i, j) * m_X.col(i);
            sum += m_z(i, j);
        }
        if (sum > 0) {
            m_ctr.col(j) = mul / sum;
        }
    }
}

void KMeans::initialize_centroids() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, m_n - 1);

    for (int j = 0; j < m_n_clusters; j++) {
        int random_index = dis(gen);
        m_ctr.col(j) = m_X.col(random_index);
    }
}
}
