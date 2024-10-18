//
// Created by 崔光远 on 2024/10/15.
//

#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <set>

#include "ClusteringBase.h"
#include "Kernels.h"

namespace mlpa::clst {
template <typename T>
class MeanShift final : public ClusteringBase {
private:
    int m_n_cur_clusters;
    T m_kernel;
    Eigen::MatrixXd m_data_points;

public:
    MeanShift() = delete;
    MeanShift(double, Eigen::MatrixXd, Eigen::RowVectorXi);
    void fit() override;
    Eigen::MatrixXd get_centers() override;
    Eigen::RowVectorXi get_labels() override;
};

template <typename T>
MeanShift<T>::MeanShift(const double h, Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : ClusteringBase(std::move(X), std::move(y)), m_n_cur_clusters(0), m_kernel(T(h)) {
    this->m_data_points = m_X; // make a copy
}

template<typename T>
void MeanShift<T>::fit() {
    for (int i = 0; i < m_X.cols(); ++i) {
        Eigen::RowVectorXd weight = this->m_kernel.calc(m_X, this->m_data_points.col(i));
        Eigen::MatrixXd mul_x = m_X.array().rowwise() * weight.array();
        this->m_data_points.col(i) = mul_x.rowwise().sum() / weight.sum();
    }
}

template<typename T>
Eigen::MatrixXd MeanShift<T>::get_centers() {
    std::vector centers(m_X.cols(), std::vector<double>(m_X.rows()));
    for (int i = 0; i < m_X.cols(); ++i) {
        for (int j = 0; j < m_X.rows(); ++j) {
            centers[i][j] = this->m_data_points(j, i);
        }
    }
    std::ranges::sort(centers);
    auto new_end = std::unique(centers.begin(), centers.end(), [] (
        const std::vector<double> & p1, const std::vector<double> & p2, const double tolerance=0.01)  {
            bool b = true;
            for (int i = 0; i < p1.size(); ++i) {
                b = b && abs(p1[i] - p2[i]) >= tolerance;
            }
            return !b;
    });
    centers.erase(new_end, centers.end());

    this->m_n_cur_clusters = static_cast<int>(centers.size());
    Eigen::MatrixXd ret(m_X.rows(), m_n_cur_clusters);
    for (int i = 0; i < m_X.rows(); ++i) {
        for (int j = 0; j < m_n_cur_clusters; ++j) {
            ret(i, j) = centers[j][i];
        }
    }
    this->m_ctr = ret;
    return ret;
}

template<typename T>
Eigen::RowVectorXi MeanShift<T>::get_labels() {
    Eigen::RowVectorXi ret(this->m_data_points.cols());
    for (int i = 0; i < m_X.cols(); i++) {
        double min_v = LONG_MAX;
        int min_i = 0;
        for (int j = 0; j < this->m_ctr.cols(); j++) {
            if (const auto diff = (m_X.col(i) - m_ctr.col(j)).norm(); diff < min_v) {
                min_v = diff;
                min_i = j;
            }
        }
        ret(i) = min_i + 1;
    }
    return ret;
}
} // namespace mlpa::clst

#endif //MEANSHIFT_H
