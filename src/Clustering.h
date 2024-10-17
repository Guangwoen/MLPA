//
// Created by 崔光远 on 2024/10/11.
//

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <iostream>
#include <Eigen/Core>

namespace mlpa {
template <typename T>
class Clustering {
private:
    int m_n_iter;
    T m_clustering;

public:
    Clustering() = delete;
    explicit Clustering(const T &);
    Clustering(const T &, int);

    /*
     * for mean-shift, third param is bandwidth
     * cluster count otherwise
     */
    Clustering(const Eigen::MatrixXd &, const Eigen::RowVectorXi &, double, int);

    void fit();
    Eigen::MatrixXd get_centers();
    Eigen::RowVectorXi get_labels();
};

template<typename T>
Clustering<T>::Clustering(const T &c): m_n_iter(1), m_clustering(c) {}

template<typename T>
Clustering<T>::Clustering(const T &c, const int i): m_n_iter(i), m_clustering(c) {}

template<typename T>
Clustering<T>::Clustering(const Eigen::MatrixXd &X, const Eigen::RowVectorXi &y,
    const double c, const int i): m_n_iter(i), m_clustering(T(c, X, y)) {}

template<typename T>
void Clustering<T>::fit() {
    for (int i = 0; i < m_n_iter; i++) {
        m_clustering.fit();
    }
}

template<typename T>
Eigen::MatrixXd Clustering<T>::get_centers() {
    return m_clustering.get_centers();
}

template<typename T>
Eigen::RowVectorXi Clustering<T>::get_labels() {
    return m_clustering.get_labels();
}
} // namespace mlpa

#endif //CLUSTERING_H
