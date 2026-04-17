#pragma once

#include <filesystem>

#include "Alias.h"

#ifndef NN_MNIST_DIR
#define NN_MNIST_DIR "external_submodules/mnist"
#endif

#ifndef NN_CIFAR10_DIR
#define NN_CIFAR10_DIR "external_submodules/CIFAR-10/cifar-10-batches-bin"
#endif

namespace nn {

struct Split {
    Matrix X_train;
    Matrix y_train;
    Matrix X_test;
    Matrix y_test;
};

class InputDataset {
   public:
    static Split LoadMnist(const std::filesystem::path& dir = std::filesystem::path{NN_MNIST_DIR});
    static Split LoadCifar10(const std::filesystem::path& dir = std::filesystem::path{NN_CIFAR10_DIR});
};

}  // namespace nn
