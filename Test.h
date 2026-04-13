#pragma once
#include <cstdint>

namespace nn {

struct TrainConfig;

struct TestConfig {
    int epochs = 15;
    int batch_size = 64;
    float lr = 0.001f;
    int train_limit = -1;
    int test_limit = -1;

    TrainConfig ToTrainConfig(std::uint64_t shuffle_seed = 42) const;
};

void RunAllTests();

void TestMnistBasic(const TestConfig& cfg);
void TestCifar10Basic(const TestConfig& cfg);

}  // namespace nn
