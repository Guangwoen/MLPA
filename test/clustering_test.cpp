//
// Created by 崔光远 on 2024/10/11.
//

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <matplot/matplot.h>

#include "../src/Clustering.h"
#include "../src/KMeans.h"

#include "../src/utils.h"

class ClusteringBaseTest : public ::testing::Test {
protected:
    static Eigen::MatrixXd XA;
    static Eigen::RowVectorXi yA;
    static Eigen::MatrixXd XB;
    static Eigen::RowVectorXi yB;
    static Eigen::MatrixXd XC;
    static Eigen::RowVectorXi yC;

protected:
    void SetUp() override {};

    void TearDown() override {};

    static void SetUpTestSuite() {
        XA = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA2/PA2-cluster-data/cluster_data_text/cluster_data_dataA_X.txt");
        yA = read_txt<int>("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA2/PA2-cluster-data/cluster_data_text/cluster_data_dataA_Y.txt");
        XB = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA2/PA2-cluster-data/cluster_data_text/cluster_data_dataB_X.txt");
        yB = read_txt<int>("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA2/PA2-cluster-data/cluster_data_text/cluster_data_dataB_Y.txt");
        XC = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA2/PA2-cluster-data/cluster_data_text/cluster_data_dataC_X.txt");
        yC = read_txt<int>("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA2/PA2-cluster-data/cluster_data_text/cluster_data_dataC_Y.txt");
    };

    static void TearDownTestSuite() {};
};

Eigen::MatrixXd ClusteringBaseTest::XA;
Eigen::RowVectorXi ClusteringBaseTest::yA;
Eigen::MatrixXd ClusteringBaseTest::XB;
Eigen::RowVectorXi ClusteringBaseTest::yB;
Eigen::MatrixXd ClusteringBaseTest::XC;
Eigen::RowVectorXi ClusteringBaseTest::yC;

constexpr static int n_clusters = 4;

constexpr static int n_iters = 200;

static void clst_plot(
    const Eigen::MatrixXd &X,
    const Eigen::MatrixXd &centers,
    const Eigen::VectorXi &label,
    const std::string& save_path) {
    const std::vector<std::string> colors = {
        "red", "yellow", "magenta", "green"
    };

    const auto f = matplot::figure(true);
    const auto xs = X.row(0);
    const auto ys = X.row(1);
    matplot::hold(matplot::on);
    for (int i = 0; i < X.cols(); ++i)
        matplot::scatter(std::vector{xs[i]}, std::vector{ys[i]})->color(colors[label[i]-1]);

    const auto cx = centers.row(0);
    const auto cy = centers.row(1);
    for (int i = 0; i < centers.cols(); ++i)
        matplot::scatter(std::vector{cx[i]}, std::vector{cy[i]})->color("black");
    matplot::hold(matplot::off);
    f->save(save_path);
}

TEST_F(ClusteringBaseTest, kmeansAClstTest) {
    mlpa::Clustering<mlpa::clst::KMeans> km(XA, yA, n_clusters, n_iters);

    km.fit();

    const auto c = km.get_centers();

    // std::cout << c << std::endl;

    const auto l = km.get_labels();

    // std::cout << l << std::endl;

    clst_plot(XA, c, l, "../../output/clustering/kmeansClstTestA.jpg");
}

TEST_F(ClusteringBaseTest, kmeansBClstTest) {
    mlpa::Clustering<mlpa::clst::KMeans> km(XB, yB, n_clusters, n_iters);

    km.fit();

    const auto c = km.get_centers();

    // std::cout << c << std::endl;

    const auto l = km.get_labels();

    // std::cout << l << std::endl;

    clst_plot(XB, c, l, "../../output/clustering/kmeansClstTestB.jpg");
}

TEST_F(ClusteringBaseTest, kmeansCClstTest) {
    mlpa::Clustering<mlpa::clst::KMeans> km(XC, yC, n_clusters, n_iters);

    km.fit();

    const auto c = km.get_centers();

    // std::cout << c << std::endl;

    const auto l = km.get_labels();

    // std::cout << l << std::endl;

    clst_plot(XC, c, l, "../../output/clustering/kmeansClstTestC.jpg");
}
