//
// Created by 崔光远 on 2024/9/25.
//

#ifndef BAYESIANREGRESSION_H
#define BAYESIANREGRESSION_H

#include "../src/RegressionBase.h"

namespace mlpa::reg {
class BayesianRegression final : public RegressionBase {
private:
    double m_alpha;
    double m_sigma_s;                                   // sigma square
    Eigen::VectorXd m_param_hat_mean;
    Eigen::MatrixXd m_param_hat_Cov;

public:
    BayesianRegression() = delete;
    ~BayesianRegression() override = default;
    BayesianRegression(Eigen::RowVectorXd, Eigen::VectorXd, double, double, unsigned);
    BayesianRegression(Eigen::RowVectorXd, Eigen::VectorXd,
        std::function<Eigen::MatrixXd(Eigen::RowVectorXd)>,
        double, double, unsigned);
    void estimate() override;
    [[nodiscard]] double get_mean_squared_error(const Eigen::RowVectorXd &, const Eigen::VectorXd &) const override;
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict_distrib(const Eigen::RowVectorXd &) const;
    [[nodiscard]] std::function<std::pair<double, double>(double)> get_predict_distrib_func() const;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> get_hat_param();
    Eigen::VectorXd predict(const Eigen::RowVectorXd &) override;                  // deleted
    [[nodiscard]] std::function<double(double)> get_predict_func() const override; // deleted
    Eigen::VectorXd get_hat_theta() override;                                      // deleted
};
} // namespace mlpa::reg

#endif //BAYESIANREGRESSION_H
