//
// Created by 崔光远 on 2024/9/25.
//

#ifndef ROBUSTREGRESSION_H
#define ROBUSTREGRESSION_H

#include "RegressionBase.h"

namespace mlpa::reg {
class RobustRegression final : public RegressionBase {
public:
    RobustRegression() = delete;
    ~RobustRegression() override = default;
    RobustRegression(Eigen::RowVectorXd, Eigen::VectorXd, unsigned);
    RobustRegression(Eigen::RowVectorXd, Eigen::VectorXd,
        std::function<Eigen::MatrixXd(Eigen::RowVectorXd)>, unsigned);
    void estimate() override;
    Eigen::VectorXd predict(const Eigen::RowVectorXd &) override;
};
} // namespace mlpa::reg

#endif //RobustRegression_H
