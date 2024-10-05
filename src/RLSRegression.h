//
// Created by 崔光远 on 2024/9/23.
//

#ifndef RLSREGRESSION_H
#define RLSREGRESSION_H

#include "RegressionBase.h"

namespace mlpa::reg {
class RLSRegression final : public RegressionBase {
private:
    double m_lambda;
public:
    ~RLSRegression() override = default;
    RLSRegression() = delete;
    RLSRegression(Eigen::RowVectorXd, Eigen::VectorXd, unsigned);
    RLSRegression(Eigen::RowVectorXd, Eigen::VectorXd, double, unsigned);
    void estimate() override;
    Eigen::VectorXd predict(const Eigen::RowVectorXd &) override;
};
} // namespace mlpa::reg

#endif //RLSREGRESSION_H
