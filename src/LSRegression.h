//
// Created by 崔光远 on 2024/9/22.
//

#ifndef LSREGRESSION_H
#define LSREGRESSION_H

#include "RegressionBase.h"

namespace mlpa::reg {
class LSRegression final : public RegressionBase {
public:
    LSRegression() = delete;
    ~LSRegression() override = default;
    LSRegression(Eigen::MatrixXd, Eigen::VectorXd, unsigned, unsigned);
    LSRegression(Eigen::MatrixXd, Eigen::VectorXd,
        std::function<Eigen::MatrixXd(Eigen::VectorXd)>, unsigned, unsigned);
    void estimate() override;
    Eigen::VectorXd predict(const Eigen::MatrixXd&) override;
};
} // namespace mlpa::reg

#endif //LSREGRESSION_H
