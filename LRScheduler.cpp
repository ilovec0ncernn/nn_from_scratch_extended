#include "LRScheduler.h"

#include <cmath>
#include <utility>

namespace nn {

LRScheduler::LRScheduler(std::function<Scalar(int)> fn) : fn_(std::move(fn)) {
}

Scalar LRScheduler::Step(int epoch) const {
    return fn_(epoch);
}

LRScheduler LRScheduler::Constant(Scalar lr) {
    return LRScheduler{[lr](int) { return lr; }};
}

LRScheduler LRScheduler::StepLR(Scalar initial_lr, Scalar gamma, int step_size) {
    return LRScheduler{[initial_lr, gamma, step_size](int epoch) {
        return initial_lr * std::pow(gamma, Scalar((epoch - 1) / step_size));
    }};
}

LRScheduler LRScheduler::ExponentialLR(Scalar initial_lr, Scalar gamma) {
    return LRScheduler{[initial_lr, gamma](int epoch) { return initial_lr * std::pow(gamma, Scalar(epoch - 1)); }};
}

}  // namespace nn
