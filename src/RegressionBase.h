//
// Created by 崔光远 on 2024/9/21.
//

#ifndef BASEML_H
#define BASEML_H

#include <functional>
#include <Eigen/Core>

namespace mlpa::reg {
class RegressionBase {
protected:
    unsigned m_n;                                               // # of sample
    unsigned m_k;                                               // dimension
    Eigen::RowVectorXd m_X;                                     // sample x
    Eigen::VectorXd m_y;                                        // sample y
    Eigen::VectorXd m_theta;                                    // init theta
    Eigen::VectorXd m_hat_theta;                                // estimated theta
    std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> m_phi;   // function phi
public:
    virtual ~RegressionBase() = default;
    RegressionBase() = delete;
    RegressionBase(Eigen::RowVectorXd, Eigen::VectorXd, unsigned);
    RegressionBase(Eigen::RowVectorXd, Eigen::VectorXd, std::function<Eigen::MatrixXd(Eigen::RowVectorXd)>, unsigned);
    RegressionBase(Eigen::RowVectorXd, Eigen::VectorXd, Eigen::VectorXd,
        std::function<Eigen::MatrixXd(Eigen::RowVectorXd)>, unsigned);
    virtual void estimate() = 0;
    virtual Eigen::VectorXd predict(const Eigen::RowVectorXd&) = 0;
    void set_phi(std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> p);
    virtual Eigen::VectorXd get_hat_theta();
    [[nodiscard]] virtual std::function<double(double)> get_predict_func() const;
    [[nodiscard]] virtual double get_mean_squared_error(const Eigen::RowVectorXd&, const Eigen::VectorXd&) const;
};
} // namespace mlpa::reg


#endif //BASEML_H
