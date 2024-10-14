//
// Created by 崔光远 on 2024/10/14.
//

#ifndef GMM_H
#define GMM_H

#include <vector>

#include <Eigen/Core>

namespace mlpa::clst {
class GMM {
private:
    int m_d;
    int m_k;
    Eigen::VectorXd m_weights;
    Eigen::MatrixXd m_mu;
    std::vector<Eigen::MatrixXd> m_cov;
public:
    GMM() = delete;
    GMM(int, int);
    GMM(int, int, const Eigen::MatrixXd &);
    [[nodiscard]] double get_one_value(const Eigen::VectorXd &, int) const;
    [[nodiscard]] double get_one_weighted_value(const Eigen::VectorXd &, int) const;
    [[nodiscard]] double get_weighted_sum(const Eigen::VectorXd &) const;
    [[nodiscard]] Eigen::VectorXd get_mu(int) const;
    void set_weight(double, int);
    void set_mu(const Eigen::VectorXd&, int);
    void set_cov(Eigen::MatrixXd, int);
};
} // namespace mlpa::clst

#endif //GMM_H
