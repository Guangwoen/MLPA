//
// Created by 崔光远 on 2024/10/16.
//

#ifndef KERNELS_H
#define KERNELS_H

#include <Eigen/Core>

#include "utils.h"

namespace mlpa::clst {
class Kernel {
protected:
    double m_h;

public:
    Kernel() = delete;
    explicit Kernel(const double h): m_h(h) {};
    virtual ~Kernel() = default;
    virtual double calc(const Eigen::VectorXd &, const Eigen::VectorXd &) = 0;
};

class GaussianKernel final : public Kernel {
public:
    GaussianKernel() = delete;
    explicit GaussianKernel(const double h): Kernel(h) {};
    double calc(const Eigen::VectorXd &X, const Eigen::VectorXd &mu) override {
        const auto d = static_cast<int>(X.rows());
        return gaussian(X, mu, Eigen::MatrixXd::Identity(d, d) * m_h * m_h);
    };
};
}

#endif //KERNELS_H
