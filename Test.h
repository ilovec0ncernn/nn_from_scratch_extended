#pragma once
#include <cstdint>

#include "Alias.h"
#include "LRScheduler.h"

namespace nn {

struct TrainConfig;

struct TestConfig {
    int epochs = 15;
    int batch_size = 64;
    float lr = 0.001f;
    Scalar weight_decay = 0.0f;
    LRScheduler scheduler = LRScheduler::Constant(0.001f);
    int train_limit = -1;
    int test_limit = -1;

    TrainConfig ToTrainConfig(std::uint64_t shuffle_seed = 42) const;
};

void RunAllTests();

void TestMnistBasic(const TestConfig& cfg);
void TestCifar10Basic(const TestConfig& cfg);
void TestCifar10CNN(const TestConfig& cfg);

}  // namespace nn
