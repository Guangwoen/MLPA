//
// Created by 崔光远 on 2024/9/21.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/LU>

int add(int a, int b);

inline double gaussian(const Eigen::VectorXd &X, const Eigen::VectorXd &mu, const Eigen::MatrixXd &cov) {
    const int d = static_cast<int>(X.rows());
    const double norm_cost = pow(2 * M_PI, - d / 2.0) * pow(cov.determinant(), -0.5);
    Eigen::VectorXd diff = X - mu;
    const double exponent = exp((-0.5 * diff.transpose() * cov.inverse() * diff).eval().value());
    return norm_cost * exponent;
}

double euclidean_distance(const Eigen::VectorXd &, const Eigen::VectorXd &);

Eigen::MatrixXd whiten(const Eigen::MatrixXd&);

double estimate_bandwidth(const Eigen::MatrixXd &, double=0.3);

template <typename T=double>
Eigen::MatrixX<T> read_txt(const std::string& path) {
    if (std::ifstream file(path); file.is_open()) {
        std::vector<std::vector<T>> lst;
        std::string line;
        while (getline(file, line)) {
            std::istringstream iss(line);
            std::string num;
            std::vector<T> nums;
            while (getline(iss, num, ' ')) {
                if (!num.empty()) {
                    nums.push_back(stod(num));
                }
            }
            lst.push_back(nums);
        }
        Eigen::MatrixX<T> ret(lst.size(), lst[0].size());
        for (int i = 0; i < lst.size(); i++)
            for (int j = 0; j < lst[0].size(); j++)
                ret(i, j) = lst[i][j];
        file.close();
        return ret;
    }
    else {
        std::cerr << "Error opening file " << path << std::endl;
    }
    return {};
}

#endif //UTILS_H
