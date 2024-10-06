//
// Created by 崔光远 on 2024/10/6.
//

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <matplot/matplot.h>

#include "../src/LASSORegression.h"
#include "../src/LSRegression.h"
#include "../src/Polynomial.h"
#include "../src/RLSRegression.h"
#include "../src/RobustRegression.h"
#include "../src/utils.h"

class CountingPeopleTest : public ::testing::Test {
protected:
    static Eigen::MatrixXd X;
    static Eigen::VectorXd y;
    static Eigen::MatrixXd test_X;
    static Eigen::VectorXd test_y;

    void SetUp() override {}

    void TearDown() override {}

    static void SetUpTestSuite() {
        std::cout << "CountingPeopleTest::SetUpTestSuite()" << std::endl;

        const auto X_ = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/count_data_trainx.txt");
        const auto y_ = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/count_data_trainy.txt");
        test_X = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/count_data_testx.txt");
        test_y = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA1/PA-1-data-text/count_data_testy.txt");

        X = X_;
        y = y_;
    }

    static void TearDownTestSuite() {
        std::cout << "CountingPeopleTest::TearDownTestSuite()" << std::endl;
    }
};

Eigen::MatrixXd CountingPeopleTest::X;
Eigen::VectorXd CountingPeopleTest::y;
Eigen::MatrixXd CountingPeopleTest::test_X;
Eigen::VectorXd CountingPeopleTest::test_y;

constexpr static unsigned order = 2;    // order of polynomial func
constexpr static unsigned k     = 9;    // original dim
constexpr static unsigned t_k   = 18;   // dim after transformation

void make_plot(
    const Eigen::VectorXd &ey,
    const Eigen::VectorXd &true_y,
    const std::string &save_path) {
    const auto f = matplot::figure(true);
    matplot::scatter(matplot::linspace(0, 600),
        std::vector(true_y.data(), true_y.data() + true_y.size()));
    matplot::hold(matplot::on);
    matplot::scatter(matplot::linspace(0, 600),
        std::vector(ey.data(), ey.data() + ey.size()));
    matplot::legend({"true value", "estimated value"});
    matplot::hold(matplot::off);
    f->save(save_path);
}

TEST_F(CountingPeopleTest, normalFeatTransLsRegTest) {
    mlpa::Polynomial<mlpa::reg::LSRegression> lr(X, y, order, k, t_k, false);

    lr.estimate();

    const auto r = std::get<Eigen::VectorXd>(lr.predict(test_X));
    const auto res = r.array().round();
    ASSERT_TRUE(res.size() != 0);

    const double mae = lr.get_mean_absolute_error(test_X, test_y);
    std::cout << "LS MAE err: " << mae << std::endl;
    const double mse = lr.get_mean_squared_error(test_X, test_y);
    std::cout << "LS MSE err: " << mse << std::endl;

    make_plot(res, test_y, "../../output/count_people/lsRegTest.jpg");
}

TEST_F(CountingPeopleTest, normalFeatTransRlsRegTest) {
    constexpr double lambda = 0.8;
    mlpa::reg::RLSRegression rl(X, y, lambda, k, t_k);
    mlpa::Polynomial rlr(order, rl, false);

    rlr.estimate();

    const auto r = std::get<Eigen::VectorXd>(rlr.predict(test_X));
    const auto res = r.array().round();
    ASSERT_TRUE(res.size() != 0);

    const double mae = rlr.get_mean_absolute_error(test_X, test_y);
    std::cout << "RLS MAE err: " << mae << std::endl;
    const double mse = rlr.get_mean_squared_error(test_X, test_y);
    std::cout << "RLS MSE err: " << mse << std::endl;

    make_plot(res, test_y, "../../output/count_people/rlsRegTest.jpg");
}

TEST_F(CountingPeopleTest, normalFeatTransLassoRegTest) {
    constexpr double lambda = 0.8;
    constexpr int nWSR = 100;
    mlpa::reg::LASSORegression lasso_r(X, y, lambda, nWSR, k, t_k);
    mlpa::Polynomial lasso(order, lasso_r, false);

    lasso.estimate();

    const auto r = std::get<Eigen::VectorXd>(lasso.predict(test_X));
    const auto res = r.array().round();
    ASSERT_TRUE(res.size() != 0);

    const double mae = lasso.get_mean_absolute_error(test_X, test_y);
    std::cout << "LASSO MAE err: " << mae << std::endl;
    const double mse = lasso.get_mean_squared_error(test_X, test_y);
    std::cout << "LASSO MSE err: " << mse << std::endl;

    make_plot(res, test_y, "../../output/count_people/lassoRegTest.jpg");
}

TEST_F(CountingPeopleTest, normalFeatTransRobustRegTest) {
    mlpa::Polynomial<mlpa::reg::RobustRegression> rr(X, y, order, k, t_k, false);

    rr.estimate();

    const auto r = std::get<Eigen::VectorXd>(rr.predict(test_X));
    const auto res = r.array().round();
    ASSERT_TRUE(res.size() != 0);

    const double mae = rr.get_mean_absolute_error(test_X, test_y);
    std::cout << "RR MAE err: " << mae << std::endl;
    const double mse = rr.get_mean_squared_error(test_X, test_y);
    std::cout << "RR MSE err: " << mse << std::endl;

    make_plot(res, test_y, "../../output/count_people/robustRegTest.jpg");
}

TEST_F(CountingPeopleTest, normalFeatTransBayesianRegTest) {
    constexpr double alpha = 0.5;
    constexpr double sigma_s = 0.04;

    mlpa::reg::BayesianRegression b(X, y, alpha, sigma_s, k, t_k);
    mlpa::Polynomial br(order, b, false);

    br.estimate();

    const auto [m, covariance]=
        std::get<std::pair<Eigen::VectorXd, Eigen::MatrixXd>>(br.predict(test_X));
    const auto mean = m.array().round();
    ASSERT_TRUE(mean.size() != 0);
    ASSERT_TRUE(covariance.size() != 0);

    const double mae = br.get_mean_absolute_error(test_X, test_y);
    std::cout << "BR MAE err: " << mae << std::endl;
    const double mse = br.get_mean_squared_error(test_X, test_y);
    std::cout << "BR MSE err: " << mse << std::endl;

    make_plot(mean, test_y, "../../output/count_people/bayesianRegTest.jpg");
}

