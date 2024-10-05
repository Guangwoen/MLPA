//
// Created by 崔光远 on 2024/9/21.
//

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <matplot/matplot.h>
#include <random>
#include <fstream>
#include <__random/random_device.h>

#include "../src/BayesianRegression.h"
#include "../src/LASSORegression.h"
#include "../src/LSRegression.h"
#include "../src/Polynomial.h"
#include "../src/RLSRegression.h"
#include "../src/RobustRegression.h"

#include "../src/utils.h"

class RegressionBaseTest : public ::testing::Test {
private:
    static auto random_elements(const Eigen::VectorXd &x, const Eigen::VectorXd& y, const long n) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<int> indices(x.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::ranges::shuffle(indices, gen);

        const std::vector<int> selected_indices(indices.begin(), indices.begin() + n);

        Eigen::RowVectorXd selected_x(n);
        Eigen::VectorXd selected_y(n);
        for (int i = 0; i < n; ++i) {
            selected_x[i] = x[selected_indices[i]];
            selected_y[i] = y[selected_indices[i]];
        }

        return std::make_pair(selected_x, selected_y);
    }
    static void add_outliers(const std::vector<std::pair<int, int>>& outliers) {
        const long size = static_cast<long>(outliers.size());
        Eigen::RowVectorXd newX(X.size() + size);
        Eigen::VectorXd newY(y.size() + size);
        newX.head(X.size()) = X;
        newY.head(y.size()) = y;
        for (int i = 0; i < size; ++i) {
            newX(i + X.size()) = outliers[i].first;
            newY(i + y.size()) = outliers[i].second;
        }
        X = newX;
        y = newY;
    }
protected:
    static Eigen::RowVectorXd X;
    static Eigen::VectorXd y;
    static Eigen::RowVectorXd Xstar;
    static Eigen::VectorXd Ystar;
    static long sample_portion;

    void SetUp() override {}

    void TearDown() override {}

    static void SetUpTestSuite() {
        std::cout << "RegressionBaseTest::SetUpSuite()" << std::endl;
        const auto X_ = read_txt<Eigen::RowVectorXd>("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/polydata_data_sampx.txt");
        const auto y_ = read_txt<Eigen::VectorXd>("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/polydata_data_sampy.txt");
        Xstar = read_txt<Eigen::RowVectorXd>("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/polydata_data_polyx.txt");
        Ystar = read_txt<Eigen::VectorXd>("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/polydata_data_polyy.txt");

        std::tie(X, y) = random_elements(X_, y_, X_.size() / (100 / sample_portion));

        // add_outliers({{-1.5, -100}, {0, 100}, {1, 50}});
    }

    static void TearDownTestSuite() {
        std::cout << "RegressionBaseTest::TearDownSuite()" << std::endl;
    }
};

Eigen::RowVectorXd RegressionBaseTest::X;
Eigen::VectorXd RegressionBaseTest::y;
Eigen::RowVectorXd RegressionBaseTest::Xstar;
Eigen::VectorXd RegressionBaseTest::Ystar;
long RegressionBaseTest::sample_portion = 100;

constexpr static unsigned k = 10;

static void reg_plot(
    const Eigen::RowVectorXd& X,
    const Eigen::VectorXd& y,
    const std::function<double(double)>& func, const std::string& save_path) {
    const auto f = matplot::figure(true);
    // const auto sp = matplot::linspace(X.minCoeff(), X.maxCoeff());
    const auto sp = matplot::linspace(-2, 2);
    matplot::plot(sp, matplot::transform(sp, [func](auto x) {
        return func(x);
    }));
    matplot::hold(matplot::on);
    matplot::scatter(std::vector(X.data(), X.data() + X.size()),
        std::vector(y.data(), y.data() + y.size()));
    matplot::hold(matplot::off);
    f->save(save_path);
}

static void bayesian_reg_plot(
    const Eigen::RowVectorXd& X,
    const Eigen::VectorXd& y,
    const auto& func, const std::string& save_path) {
    const auto f = matplot::figure(true);
    // const auto sp = matplot::linspace(X.minCoeff(), X.maxCoeff());
    const auto sp = matplot::linspace(-2, 2);
    matplot::errorbar(sp, matplot::transform(sp, [func](auto x) {
        return func(x).first;
    }), matplot::transform(sp, [func](auto x) {
        return sqrt(func(x).second);
    }))->filled_curve(true);
    matplot::hold(matplot::on);
    matplot::scatter(std::vector(X.data(), X.data() + X.size()),
        std::vector(y.data(), y.data() + y.size()));
    matplot::hold(matplot::off);
    f->save(save_path);
}

TEST_F(RegressionBaseTest, lsRegTest) {
    mlpa::Polynomial<mlpa::reg::LSRegression> lr(X, y, k);

    lr.estimate();

    const auto res = std::get<Eigen::VectorXd>(lr.predict(Xstar));
    ASSERT_TRUE(res.size() != 0);

    const auto func = std::get<std::function<double(double)>>(lr.get_predict_func());
    reg_plot(X, y, func, "../../output/lsRegTest" + std::to_string(sample_portion) + ".jpg");
    const auto err = lr.get_mean_squared_error(Xstar, Ystar);
    // std::cout << "Least-Square Regression mean square error: " << err << std::endl;

    std::fstream f;
    f.open("../../output/ls-mse.txt",std::ios::out | std::ios::app);
    f << sample_portion << " " << err <<std::endl;
    f.close();

    std::cout << std::get<Eigen::VectorXd>(lr.get_estimated_param()) << std::endl;
}

TEST_F(RegressionBaseTest, rlsRegTest) {
    constexpr double lambda = 0.8;
    mlpa::reg::RLSRegression rl(X, y, lambda, k);
    mlpa::Polynomial rlr(k, rl);

    rlr.estimate();

    const auto res = std::get<Eigen::VectorXd>(rlr.predict(Xstar));
    ASSERT_TRUE(res.size() != 0);

    const auto func = std::get<std::function<double(double)>>(rlr.get_predict_func());
    reg_plot(X, y, func, "../../output/rlsRegTest" + std::to_string(sample_portion) + ".jpg");
    const auto err = rlr.get_mean_squared_error(Xstar, Ystar);
    // std::cout << "Regularized LS Regression mean square error: " << err << std::endl;

    std::fstream f;
    f.open("../../output/rls-mse.txt",std::ios::out | std::ios::app);
    f << sample_portion << " " << err <<std::endl;
    f.close();
}

TEST_F(RegressionBaseTest, lassoRegTest) {
    constexpr double lambda = 0.8;
    constexpr int nWSR = 100;
    mlpa::reg::LASSORegression lasso_r(X, y, lambda, nWSR, k);
    mlpa::Polynomial lasso(k, lasso_r);

    lasso.estimate();

    const auto res = std::get<Eigen::VectorXd>(lasso.predict(Xstar));
    ASSERT_TRUE(res.size() != 0);

    const auto func = std::get<std::function<double(double)>>(lasso.get_predict_func());
    reg_plot(X, y, func, "../../output/lassoRegTest" + std::to_string(sample_portion) + ".jpg");
    const auto err = lasso.get_mean_squared_error(Xstar, Ystar);
    std::cout << "LASSO mean square error: " << err << std::endl;

    std::fstream f;
    f.open("../../output/lasso-mse.txt",std::ios::out | std::ios::app);
    f << sample_portion << " " << err <<std::endl;
    f.close();
}

TEST_F(RegressionBaseTest, robustRegTest) {
    mlpa::Polynomial<mlpa::reg::RobustRegression> rr(X, y, k);

    rr.estimate();

    const auto res = std::get<Eigen::VectorXd>(rr.predict(Xstar));
    ASSERT_TRUE(res.size() != 0);

    const auto func = std::get<std::function<double(double)>>(rr.get_predict_func());
    reg_plot(X, y, func, "../../output/robustRegTest" + std::to_string(sample_portion) + ".jpg");
    const auto err = rr.get_mean_squared_error(Xstar, Ystar);
    std::cout << "Robust Regression mean square error: " << err << std::endl;

    std::fstream f;
    f.open("../../output/rr-mse.txt",std::ios::out | std::ios::app);
    f << sample_portion << " " << err <<std::endl;
    f.close();
}

TEST_F(RegressionBaseTest, bayesianRegTest) {
    constexpr double alpha = 0.5;
    constexpr double sigma_s = 0.04;
    mlpa::reg::BayesianRegression b(X, y, alpha, sigma_s, k);
    mlpa::Polynomial br(k, b);

    br.estimate();

    const auto [mean, covariance]
        = std::get<std::pair<Eigen::VectorXd, Eigen::MatrixXd>>(br.predict(Xstar));
    ASSERT_TRUE(mean.size() != 0);
    ASSERT_TRUE(covariance.size() != 0);

    const auto func
        = std::get<std::function<std::pair<double, double>(double)>>(br.get_predict_func());
    bayesian_reg_plot(X, y, func, "../../output/bayesianRegTest" + std::to_string(sample_portion) + ".jpg");
    const auto err = br.get_mean_squared_error(Xstar, Ystar);
    std::cout << "Bayesian Regression mean square error: " << err << std::endl;

    std::fstream f;
    f.open("../../output/bayesian-mse.txt",std::ios::out | std::ios::app);
    f << sample_portion << " " << err <<std::endl;
    f.close();
}