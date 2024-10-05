//
// Created by 崔光远 on 2024/9/21.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <Eigen/Core>

int add(int a, int b);

template <class T>
T read_txt(const std::string& path) {
    if (std::ifstream file(path); file.is_open()) {
        std::vector<double> nums;
        double item = 0.0;
        while (file >> item) nums.push_back(item);
        T ret(nums.size());
        for (auto i = 0; i < nums.size(); i++) ret(i) = nums[i];
        file.close();
        return ret;
    }
    else {
        std::cerr << "Error opening file " << path << std::endl;
    }
    return T();
}

void make_plot(const auto& Xs, const auto& ys, const std::string& output_path);

#endif //UTILS_H
