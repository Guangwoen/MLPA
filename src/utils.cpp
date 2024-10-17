//
// Created by 崔光远 on 2024/9/21.
//

#include <iostream>

#include "utils.h"

int add(const int a, const int b) {
    return a + b;
}

double gaussian(const Eigen::VectorXd &X, const Eigen::VectorXd &mu, const Eigen::MatrixXd &cov) {
    const int d = static_cast<int>(X.rows());
    const double norm_cost = pow(2 * M_PI, - d / 2.0) * pow(cov.determinant(), -0.5);
    Eigen::VectorXd diff = X - mu;
    const double exponent = exp((-0.5 * diff.transpose() * cov.inverse() * diff).eval().value());
    return norm_cost * exponent;
}
