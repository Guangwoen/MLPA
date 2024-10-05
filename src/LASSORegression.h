//
// Created by 崔光远 on 2024/9/23.
//

#ifndef LASSOREGRESSION_H
#define LASSOREGRESSION_H

#include "../src/RegressionBase.h"

namespace mlpa::reg {
class LASSORegression final : public RegressionBase {
private:
    int m_nWSR;
    double m_lambda;
public:
    LASSORegression() = delete;
    ~LASSORegression() override = default;
    LASSORegression(Eigen::MatrixXd, Eigen::VectorXd, unsigned, unsigned);
    LASSORegression(Eigen::MatrixXd, Eigen::VectorXd, double, unsigned, unsigned);
    LASSORegression(Eigen::MatrixXd, Eigen::VectorXd, double, int, unsigned, unsigned);
    void estimate() override;
    Eigen::VectorXd predict(const Eigen::MatrixXd &) override;
};
} // namespace mlpa::reg

#endif //LASSOREGRESSION_H
