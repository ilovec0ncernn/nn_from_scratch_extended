#include "Test.h"

#include <iomanip>
#include <iostream>
#include <utility>

#include "ActivationFunctions.h"
#include "Alias.h"
#include "InputDataset.h"
#include "LossFunctions.h"
#include "Metrics.h"
#include "Network.h"

namespace nn {

TrainConfig TestConfig::ToTrainConfig(std::uint64_t shuffle_seed) const {
    TrainConfig t;
    t.epochs = epochs;
    t.batch_size = batch_size;
    t.lr = lr;
    t.shuffle_seed = shuffle_seed;
    return t;
}

static Matrix TakeCols(const Matrix& M, int n) {
    if (n < 0 || n >= M.cols()) {
        return M;
    }
    return M.leftCols(n);
}

static Split LoadData(const TestConfig& cfg) {
    Split s = InputDataset::LoadMnist();
    s.X_train = TakeCols(s.X_train, cfg.train_limit);
    s.y_train = TakeCols(s.y_train, cfg.train_limit);
    s.X_test = TakeCols(s.X_test, cfg.test_limit);
    s.y_test = TakeCols(s.y_test, cfg.test_limit);
    return s;
}

static Network CreateNet(RNG& rng) {
    Network net;
    net.AddFirstLayer(784, 128, Activation::ReLU(), rng).AddLayer(10, Activation::Identity(), rng);
    return net;
}

static void TrainNet(Network& net, const Split& s, const TestConfig& cfg) {
    TrainConfig tcfg;
    tcfg.epochs = cfg.epochs;
    tcfg.batch_size = cfg.batch_size;
    tcfg.lr = cfg.lr;

    Loss loss = Loss::CrossEntropy();
    net.Train(s.X_train, s.y_train, s.X_test, s.y_test, tcfg, loss);
}

void TestMnistBasic(const TestConfig& cfg) {
    std::cout << "training model on mnist dataset\n";

    RNG rng;
    Split data = LoadData(cfg);
    Network net = CreateNet(rng);

    TrainNet(net, data, cfg);

    Metric acc = Metric::Accuracy();
    Metric ce = Metric::CrossEntropy();
    Matrix logits = net.Predict(data.X_test);
    Scalar final_acc = acc.Value(data.y_test, logits);
    Scalar final_ce = ce.Value(data.y_test, logits);

    std::cout << "[test] final test accuracy=" << std::fixed << std::setprecision(4) << final_acc
              << ", final CE=" << std::fixed << std::setprecision(4) << final_ce << std::endl;
}

void RunAllTests() {
    TestConfig cfg;
    TestMnistBasic(cfg);
}

}  // namespace nn
