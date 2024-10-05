//
// Created by 崔光远 on 2024/9/21.
//

#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include "BayesianRegression.h"
#include "RegressionBase.h"

namespace mlpa {
template <typename T>
class Polynomial {
private:
    unsigned m_order;
    T m_regression;
    std::function<Eigen::MatrixXd(Eigen::RowVectorXd)> m_phi;

private:
    void set_phi();

public:
    Polynomial() = delete;
    Polynomial(const Eigen::RowVectorXd &, const Eigen::VectorXd &, unsigned);
    Polynomial(unsigned, T &);
    ~Polynomial() = default;
    void estimate();
    [[nodiscard]] std::variant<std::pair<Eigen::VectorXd, Eigen::MatrixXd>, Eigen::VectorXd>
    predict(const Eigen::RowVectorXd &);
    [[nodiscard]] std::variant<std::function<std::pair<double, double>(double)>, std::function<double(double)>>
    get_predict_func();
    [[nodiscard]] double get_mean_squared_error(const Eigen::RowVectorXd &, const Eigen::VectorXd &);
    std::variant<std::pair<Eigen::VectorXd, Eigen::MatrixXd>, Eigen::VectorXd> get_estimated_param();
};

template<typename T>
Polynomial<T>::Polynomial(const Eigen::RowVectorXd &X, const Eigen::VectorXd &y, const unsigned k)
    : m_order(k), m_regression(T(X, y, k)) {
    this->set_phi();
}

template<typename T>
Polynomial<T>::Polynomial(const unsigned k, T &reg)
    : m_order(k), m_regression(reg) {
    this->set_phi();
}

template <typename T>
void Polynomial<T>::set_phi() {
    this->m_phi = [this] (const Eigen::RowVectorXd& x) -> Eigen::MatrixXd {
        Eigen::MatrixXd retmat(this->m_order + 1, x.size());
        for (auto i = 0; i < x.size(); i++) {
            Eigen::VectorXd col(this->m_order + 1);
            for (int j = 0; j <= this->m_order; j++) col(j) = pow(x(i), j);
            retmat.col(i) = col;
        }
        return retmat;
    };
    this->m_regression.set_phi(this->m_phi);
}

template<typename T>
void Polynomial<T>::estimate() {
    this->m_regression.estimate();
}

template<typename T>
std::variant<std::pair<Eigen::VectorXd, Eigen::MatrixXd>, Eigen::VectorXd>
    Polynomial<T>::predict(const Eigen::RowVectorXd &X_star) {
    if constexpr (std::is_same_v<T, reg::BayesianRegression>)
        return this->m_regression.predict_distrib(X_star);
    return this->m_regression.predict(X_star);
}

template<typename T>
std::variant<std::function<std::pair<double, double>(double)>, std::function<double(double)>>
    Polynomial<T>::get_predict_func() {
    if constexpr (std::is_same_v<T, reg::BayesianRegression>)
        return this->m_regression.get_predict_distrib_func();
    return this->m_regression.get_predict_func();
}

template<typename T>
double Polynomial<T>::get_mean_squared_error(const Eigen::RowVectorXd &X, const Eigen::VectorXd &y) {
    return this->m_regression.get_mean_squared_error(X, y);
}

template<typename T>
std::variant<std::pair<Eigen::VectorXd, Eigen::MatrixXd>, Eigen::VectorXd> Polynomial<T>::get_estimated_param() {
    if constexpr (std::is_same_v<T, reg::BayesianRegression>)
        return this->m_regression.get_hat_param();
    return this->m_regression.get_hat_theta();
}

} // namespace mlpa

#endif //POLYNOMIAL_H
