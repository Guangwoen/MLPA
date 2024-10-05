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
    LASSORegression(Eigen::RowVectorXd, Eigen::VectorXd, unsigned);
    LASSORegression(Eigen::RowVectorXd, Eigen::VectorXd, double, unsigned);
    LASSORegression(Eigen::RowVectorXd, Eigen::VectorXd, double, int, unsigned);
    void estimate() override;
    Eigen::VectorXd predict(const Eigen::RowVectorXd &) override;
};
} // namespace mlpa::reg

#endif //LASSOREGRESSION_H
