//
// Created by 崔光远 on 2024/10/11.
//

#include <gtest/gtest.h>
#include <Eigen/Core>

class ClusteringBaseTest : public ::testing::Test {
protected:
    static Eigen::MatrixXd X;
    static Eigen::VectorXd y;

protected:
    void SetUp() override {};

    void TearDown() override {};

    static void SetUpTestSuite() {};

    static void TearDownTestSuite() {};
};

Eigen::MatrixXd ClusteringBaseTest::X;
Eigen::VectorXd ClusteringBaseTest::y;
