//
// Created by 崔光远 on 2024/10/16.
//

#ifndef KERNELS_H
#define KERNELS_H

#include <Eigen/Core>

#include "utils.h"

namespace mlpa::clst {
class GaussianKernel final {
private:
    double m_h;
public:
    GaussianKernel() = delete;
    explicit GaussianKernel(const double h): m_h(h) {};
    [[nodiscard]] Eigen::RowVectorXd calc(const Eigen::MatrixXd &X, const Eigen::VectorXd &mu) const {
        const auto distance = (X.colwise() - mu).colwise().squaredNorm().eval();
        // return (1 / (m_h * sqrt(2 * M_PI))) * exp(-0.5 * (distance / m_h).array().pow(2));
        return exp(-0.5 * (distance / m_h).array().pow(2));
    };
};
}

#endif //KERNELS_H
