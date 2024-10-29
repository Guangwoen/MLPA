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

const std::string ImageSegTest::image_name = "310007";
Eigen::MatrixXd ImageSegTest::image;

constexpr static int n_clusters = 3;

constexpr static int max_iter = 350;

constexpr static double bandwidth = 79.2;

static void seg_plot(const Eigen::RowVectorXi &label, const std::string &image_name, const std::string &method) {
    std::ofstream out;
    out.open("/Users/cuiguangyuan/Documents/CityU/SemesterA/Machine Learning/MLPA/output/image_segmentation/labels/" + image_name + "-" + method + ".txt");
    out << label.array();
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

    // const auto estimated_bandwidth = estimate_bandwidth(image, 0.05);
    //
    // std::cout << estimated_bandwidth << std::endl;

    // std::cout << image << std::endl;

    mlpa::Clustering<mlpa::clst::MeanShift<mlpa::clst::GaussianKernel>> ms(image, {}, bandwidth);

    ms.fit(max_iter);

    const auto l = ms.get_labels();

    seg_plot(l, image_name, "Meanshift");
}
