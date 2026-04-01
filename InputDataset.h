#pragma once

#include <filesystem>

#include "Alias.h"

#ifndef NN_MNIST_DIR
#define NN_MNIST_DIR "external_submodules/mnist"
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
};

}  // namespace nn
