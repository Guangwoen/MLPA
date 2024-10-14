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
    void fit() override;
    void cluster_assignment();
    void estimate_center();
};
} // namespace mlpa::clst

#endif //KMEANS_H
