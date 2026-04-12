#pragma once

#include <vector>

#include "Alias.h"

namespace nn {

struct TrainHistory {
    std::vector<Scalar> train_acc;
    std::vector<Scalar> val_acc;
    std::vector<Scalar> val_ce;
};

}  // namespace nn
