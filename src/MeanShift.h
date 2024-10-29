//
// Created by 崔光远 on 2024/10/15.
//

#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <thread>
#include <mutex> // Include the mutex header

#include "ClusteringBase.h"
#include "Kernels.h"

namespace mlpa::clst {
template <typename T>
class MeanShift final : public ClusteringBase {
private:
    int m_n_cur_clusters;
    T m_kernel;
    Eigen::MatrixXd m_data_points;
    std::mutex mtx;

public:
    MeanShift() = delete;
    MeanShift(double, Eigen::MatrixXd, Eigen::RowVectorXi);
    void fit(int) override;
    Eigen::MatrixXd get_centers() override;
    Eigen::RowVectorXi get_labels() override;
};

template <typename T>
MeanShift<T>::MeanShift(const double h, Eigen::MatrixXd X, Eigen::RowVectorXi y)
    : ClusteringBase(std::move(X), std::move(y)), m_n_cur_clusters(0), m_kernel(T(h)) {
    this->m_data_points = m_X; // make a copy
}

template<typename T>
void MeanShift<T>::fit(const int max_iter) {
    Eigen::initParallel();
    bool has_converged = false;
    int iter = 0;
    while (!has_converged && iter < max_iter) {
        // std::cout << "Iteration " << iter << std::endl;
        iter++;

        std::vector<std::thread> threads; // Create a vector to hold threads
        for (int i = 0; i < this->m_data_points.cols(); ++i) {
            threads.emplace_back([&, i] {
                Eigen::RowVectorXd weight = this->m_kernel.calc(m_X, this->m_data_points.col(i));
                Eigen::MatrixXd mul_x = m_X.array().rowwise() * weight.array();
                if (weight.sum() == 0) return;
                std::lock_guard lock(mtx);
                this->m_data_points.col(i) = mul_x.rowwise().sum() / weight.sum();
            });
        }
        for (auto& t : threads) t.join();

        auto old_ctrs = this->m_ctr;
        auto new_ctrs = get_centers();
        this->m_ctr = new_ctrs;
        if (old_ctrs.cols() != new_ctrs.cols()) continue; // for the first iteration
        has_converged = (old_ctrs - new_ctrs).norm() < C_TOLERANCE;
    }
}

template<typename T>
Eigen::MatrixXd MeanShift<T>::get_centers() {
    auto matrix = this->m_data_points;

    std::vector<Eigen::VectorXd> centers(matrix.cols());
    centers.assign(matrix.colwise().begin(), matrix.colwise().end());
    sort(centers.begin(), centers.end(), [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a.norm() < b.norm();
    });
    auto new_end = std::unique(centers.begin(), centers.end(), [&] (
        const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
        return (a - b).norm() < 1e-5;
    });
    centers.erase(new_end, centers.end());

    this->m_n_cur_clusters = static_cast<int>(centers.size());
    Eigen::MatrixXd ret(matrix.rows(), m_n_cur_clusters);
    for (int i = 0; i < centers.size(); i++) {
        ret.col(i) = centers[i];
    }
    return ret;
}

template<typename T>
Eigen::RowVectorXi MeanShift<T>::get_labels() {
    Eigen::RowVectorXi ret(this->m_data_points.cols());
    for (int i = 0; i < m_X.cols(); i++) {
        double min_v = std::numeric_limits<double>::max();
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
