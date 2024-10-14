//
// Created by 崔光远 on 2024/10/8.
//

#ifndef CLUSTERINGBASE_H
#define CLUSTERINGBASE_H

#include <Eigen/Core>

namespace mlpa::clst {
class ClusteringBase {
protected:
    int m_n_clusters;                               // (k)
    long m_n;                                       // (n)
    Eigen::MatrixXd     m_X;                        // data points (d, n)
    Eigen::MatrixXd     m_ctr;                      // centers (d, k)
    Eigen::MatrixXd     m_z;                        // cluster assignments
    Eigen::RowVectorXi  m_y;                        // true labels
    Eigen::RowVectorXi  m_estimated_y;              // estimated labels

public:
    ClusteringBase() = delete;
    ClusteringBase(Eigen::MatrixXd, Eigen::RowVectorXi);
    ClusteringBase(int, Eigen::MatrixXd, Eigen::RowVectorXi);
    virtual ~ClusteringBase() = default;
    virtual void fit() = 0;
    virtual Eigen::MatrixXd get_centers();
    virtual Eigen::RowVectorXi get_labels();
};
} // namespace mlpa::clst

#endif //CLUSTERINGBASE_H
