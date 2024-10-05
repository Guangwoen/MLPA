#include "src/utils.h"
#include <vector>
#include <matplot/matplot.h>
#include <set>

int main()
{
    std::vector<std::string> paths = {
        "../output/bayesian-mse",
        "../output/lasso-mse",
        "../output/ls-mse",
        "../output/rls-mse",
        "../output/rr-mse"
    };
    for (const std::string& path : paths) {
        if (std::ifstream f(path + ".txt"); f.is_open()) {
            std::set<int> ps;
            std::unordered_map<int, int> ps_count;
            std::unordered_map<int, double> mse;

            int portion = 0;
            double error = 0.0;
            while (f >> portion >> error) {
                ps.insert(portion);
                ps_count[portion]++;
                mse[portion] += error;
            }
            f.close();

            std::vector<double> errs;
            errs.reserve(ps.size());
            for (const auto& p : ps) {
                errs.push_back(mse[p] / ps_count[p]);
            }

            const auto fig = matplot::figure(true);
            matplot::plot(std::vector(ps.begin(), ps.end()), errs);
            matplot::xlabel("Portion");
            matplot::ylabel("Error");

            fig->save(path + "-plot.jpg");
        }
        else std::cerr << "Unable to open file" << std::endl;
    }
    return 0;
}
