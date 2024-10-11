//
// Created by 崔光远 on 2024/10/11.
//

#ifndef KMEANS_H
#define KMEANS_H
#include "ClusteringBase.h"

namespace mlpa::clst {
class KMeans final : public ClusteringBase {
public:
    KMeans() = delete;
    KMeans(Eigen::MatrixXd, Eigen::RowVectorXi);
    KMeans(int, Eigen::MatrixXd, Eigen::RowVectorXi);
    ~KMeans() override = default;
    void cluster_assignment() override;
    void estimate_center() override;
};
} // namespace mlpa::clst

#endif //KMEANS_H
