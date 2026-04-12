#pragma once

#include <functional>

#include "Alias.h"

namespace nn {

class LRScheduler {
   public:
    LRScheduler() = default;
    explicit LRScheduler(std::function<Scalar(int)> fn);

    Scalar Step(int epoch) const;

    static LRScheduler Constant(Scalar lr);
    static LRScheduler StepLR(Scalar initial_lr, Scalar gamma, int step_size);
    static LRScheduler ExponentialLR(Scalar initial_lr, Scalar gamma);

   private:
    std::function<Scalar(int)> fn_;
};

}  // namespace nn
