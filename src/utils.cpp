//
// Created by 崔光远 on 2024/9/21.
//

#include <iostream>

#include "utils.h"

int add(const int a, const int b) {
    return a + b;
}

// 计算欧氏距离
double euclidean_distance(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    return (a - b).norm();
}

Eigen::MatrixXd whiten(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd whitened_matrix = matrix;
    for (int i = 0; i < matrix.rows(); ++i) {
        if (double stddev = std::sqrt((matrix.row(i).array() - matrix.row(i).mean()).square().sum() / (matrix.cols() - 1)); stddev > 0) {
            whitened_matrix.row(i) = matrix.row(i) / stddev;
        }
    }
    return whitened_matrix;
}

// 估计带宽
double estimate_bandwidth(const Eigen::MatrixXd &X, const double quantile) {
    const auto n_samples = X.cols();
    std::vector<double> distances;

    // 计算所有样本点之间的距离
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            if (i != j) {
                distances.push_back(euclidean_distance(X.col(i), X.col(j)));
            }
        }
    }

    // 对距离进行排序
    std::ranges::sort(distances);

    // 计算带宽（这里使用距离的指定分位数）
    const int quantile_index = static_cast<int>(quantile * static_cast<double>(distances.size()));
    return distances[quantile_index];
}
