//
// Created by 崔光远 on 2024/10/25.
//

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <matplot/matplot.h>

#include "../src/Clustering.h"
#include "../src/EM.h"
#include "../src/GMM.h"
#include "../src/KMeans.h"
#include "../src/MeanShift.h"

#include "../src/utils.h"

class ImageSegTest: public ::testing::Test {
protected:
    static const std::string image_name;
    static Eigen::MatrixXd image;

protected:
    void SetUp() override {};

    void TearDown() override {};

    static void SetUpTestSuite() {
        image = read_txt("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/Programming Assignments/PA2/images_txt/" + image_name + "_X.txt");
    };

    static void TearDownTestSuite() {};
};

const std::string ImageSegTest::image_name = "62096";
Eigen::MatrixXd ImageSegTest::image;

constexpr static int n_clusters = 5;

constexpr static int max_iter = 350;

constexpr static double bandwidth = 14.688657407491657;

static void seg_plot(const Eigen::RowVectorXi &label, const std::string &image_name, const std::string &method) {
    std::ofstream out;
    out.open("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/MLPA/output/image_segmentation/labels/" + image_name + "-" + method + ".txt");
    out << label;
    out.close();
}

TEST_F(ImageSegTest, kmeansImgSegTest) {
    image = whiten(image);
    mlpa::Clustering<mlpa::clst::KMeans> km(image, {}, n_clusters);

    km.fit();

    const auto l = km.get_labels();

    seg_plot(l, image_name, "Kmeans");
}

TEST_F(ImageSegTest, emGmmImgSegTest) {
    image = whiten(image);
    mlpa::Clustering<mlpa::clst::EM<mlpa::clst::GMM>> em(image, {}, n_clusters);

    em.fit();

    const auto l = em.get_labels();

    seg_plot(l, image_name, "EmGmm");
}

TEST_F(ImageSegTest, msImgSegTest) {

    mlpa::Clustering<mlpa::clst::MeanShift<mlpa::clst::GaussianKernel>> ms(image, {}, bandwidth);

    ms.fit(max_iter);

    const auto l = ms.get_labels();

    seg_plot(l, image_name, "Meanshift");
}

TEST_F(ImageSegTest, kmeansWeightedImgSegTest) {
    constexpr double lambda = 0.5;
    image = whiten(image);
    image.row(0) *= lambda;
    image.row(1) *= lambda;
    image.row(2) *= 1 - lambda;
    image.row(3) *= 1 - lambda;
    mlpa::Clustering<mlpa::clst::KMeans> km(image, {}, n_clusters);

    km.fit();

    const auto l = km.get_labels();

    seg_plot(l, image_name, "Kmeans");
}

TEST_F(ImageSegTest, msWeightedImgSegTest) {
    constexpr double lambda = 0.5;
    image.row(0) *= lambda;
    image.row(1) *= lambda;
    image.row(2) *= 1 - lambda;
    image.row(3) *= 1 - lambda;
    mlpa::Clustering<mlpa::clst::MeanShift<mlpa::clst::GaussianKernel>> ms(image, {}, bandwidth);

    ms.fit(max_iter);

    const auto l = ms.get_labels();

    seg_plot(l, image_name, "Meanshift");
}
