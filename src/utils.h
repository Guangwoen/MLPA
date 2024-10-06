//
// Created by 崔光远 on 2024/9/21.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <Eigen/Core>

int add(int a, int b);

inline Eigen::MatrixXd read_txt(const std::string& path) {
    if (std::ifstream file(path); file.is_open()) {
        std::vector<std::vector<double>> lst;
        std::string line;
        while (getline(file, line)) {
            std::istringstream iss(line);
            std::string num;
            std::vector<double> nums;
            while (getline(iss, num, ' ')) {
                if (!num.empty()) {
                    nums.push_back(stod(num));
                }
            }
            lst.push_back(nums);
        }
        Eigen::MatrixXd ret(lst.size(), lst[0].size());
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
