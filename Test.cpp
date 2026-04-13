#include "Test.h"

#include <iomanip>
#include <iostream>
#include <utility>

#include "ActivationFunctions.h"
#include "Alias.h"
#include "InputDataset.h"
#include "LossFunctions.h"
#include "LRScheduler.h"
#include "Metrics.h"
#include "Network.h"
#include "Optimizer.h"
#include "TrainHistory.h"
#include "WeightInit.h"

namespace nn {

TrainConfig TestConfig::ToTrainConfig(std::uint64_t shuffle_seed) const {
    TrainConfig t;
    t.epochs = epochs;
    t.batch_size = batch_size;
    t.lr = lr;
    t.shuffle_seed = shuffle_seed;
    t.scheduler = LRScheduler::Constant(lr);
    return t;
}

static Matrix TakeCols(const Matrix& M, int n) {
    if (n < 0 || n >= M.cols()) {
        return M;
    }
    return M.leftCols(n);
}

static Split LoadMnistData(const TestConfig& cfg) {
    Split s = InputDataset::LoadMnist();
    s.X_train = TakeCols(s.X_train, cfg.train_limit);
    s.y_train = TakeCols(s.y_train, cfg.train_limit);
    s.X_test = TakeCols(s.X_test, cfg.test_limit);
    s.y_test = TakeCols(s.y_test, cfg.test_limit);
    return s;
}

static Split LoadCifarData(const TestConfig& cfg) {
    Split s = InputDataset::LoadCifar10();
    s.X_train = TakeCols(s.X_train, cfg.train_limit);
    s.y_train = TakeCols(s.y_train, cfg.train_limit);
    s.X_test = TakeCols(s.X_test, cfg.test_limit);
    s.y_test = TakeCols(s.y_test, cfg.test_limit);
    return s;
}

static Network CreateMnistNet(RNG& rng) {
    Network net;
    net.AddFirstLayer(784, 128, Activation::ReLU(), rng, WeightInit::He, Optimizer::Adam(0.001f))
        .AddLayer(10, Activation::Identity(), rng, WeightInit::Xavier, Optimizer::Adam(0.001f));
    return net;
}

static Network CreateCifarNet(RNG& rng) {
    Network net;
    net.AddFirstLayer(3072, 512, Activation::ReLU(), rng, WeightInit::He, Optimizer::Adam(0.001f))
        .AddDropout(0.3f)
        .AddLayer(256, Activation::ReLU(), rng, WeightInit::He, Optimizer::Adam(0.001f))
        .AddDropout(0.3f)
        .AddLayer(10, Activation::Identity(), rng, WeightInit::Xavier, Optimizer::Adam(0.001f));
    return net;
}

static TrainHistory TrainNet(Network& net, const Split& s, const TestConfig& cfg) {
    TrainConfig tcfg;
    tcfg.epochs = cfg.epochs;
    tcfg.batch_size = cfg.batch_size;
    tcfg.lr = cfg.lr;
    tcfg.shuffle_seed = 42;
    tcfg.scheduler = LRScheduler::Constant(cfg.lr);

    Loss loss = Loss::CrossEntropy();
    return net.Train(s.X_train, s.y_train, s.X_test, s.y_test, tcfg, loss);
}

static void PrintResults(const TrainHistory& history, const Matrix& logits, const Matrix& y_test) {
    Scalar final_acc = Metric::Accuracy().Value(y_test, logits);
    Scalar final_ce = Metric::CrossEntropy().Value(y_test, logits);
    Scalar final_precision = Metric::Precision().Value(y_test, logits);
    Scalar final_recall = Metric::Recall().Value(y_test, logits);

    std::cout << "[test] accuracy=" << std::fixed << std::setprecision(4) << final_acc << "  CE=" << std::fixed
              << std::setprecision(4) << final_ce << "  precision=" << std::fixed << std::setprecision(4)
              << final_precision << "  recall=" << std::fixed << std::setprecision(4) << final_recall << std::endl;

    std::cout << "[history] val_acc per epoch:";
    for (int i = 0; i < static_cast<int>(history.val_acc.size()); ++i)
        std::cout << " " << std::fixed << std::setprecision(4) << history.val_acc[i];
    std::cout << std::endl;
}

void TestMnistBasic(const TestConfig& cfg) {
    std::cout << "training model on mnist dataset\n";

    RNG rng;
    Split data = LoadMnistData(cfg);
    Network net = CreateMnistNet(rng);
    TrainHistory history = TrainNet(net, data, cfg);

    Matrix logits = net.Predict(data.X_test);
    PrintResults(history, logits, data.y_test);
}

void TestCifar10Basic(const TestConfig& cfg) {
    std::cout << "training model on cifar-10 dataset\n";

    RNG rng;
    Split data = LoadCifarData(cfg);
    Network net = CreateCifarNet(rng);
    TrainHistory history = TrainNet(net, data, cfg);

    Matrix logits = net.Predict(data.X_test);
    PrintResults(history, logits, data.y_test);
}

void RunAllTests() {
    TestConfig cfg;
    TestMnistBasic(cfg);
    TestCifar10Basic(cfg);
}

}  // namespace nn
