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
    unsigned m_k;                                               // # of feature (d)
    unsigned m_t_k;                                             // dim after transformation (D)
    Eigen::MatrixXd m_X;                                        // sample x (d rows, n cols)
    Eigen::VectorXd m_y;                                        // sample y (n rows)
    Eigen::VectorXd m_theta;                                    // init theta (D rows)
    Eigen::VectorXd m_hat_theta;                                // estimated theta (D rows)
    std::function<Eigen::MatrixXd(Eigen::VectorXd)> m_phi;      // function phi
public:
    virtual ~RegressionBase() = default;
    RegressionBase() = delete;
    RegressionBase(Eigen::MatrixXd, Eigen::VectorXd, unsigned, unsigned);
    RegressionBase(Eigen::MatrixXd, Eigen::VectorXd,
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)>, unsigned, unsigned);
    RegressionBase(Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
        std::function<Eigen::MatrixXd(Eigen::VectorXd)>, unsigned, unsigned);
    virtual void estimate() = 0;
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd&) = 0;
    void set_phi(std::function<Eigen::MatrixXd(Eigen::VectorXd)> p);
    virtual Eigen::VectorXd get_hat_theta();
    [[nodiscard]] Eigen::MatrixXd transform(const Eigen::MatrixXd &) const;
    [[nodiscard]] virtual std::function<double(Eigen::VectorXd)> get_predict_func() const;
    [[nodiscard]] virtual double get_mean_squared_error(const Eigen::MatrixXd&, const Eigen::VectorXd&) const;
    [[nodiscard]] unsigned get_t_k() const;
};
} // namespace mlpa::reg


#endif //BASEML_H
