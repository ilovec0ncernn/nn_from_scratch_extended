#pragma once

#include <functional>

#include "Alias.h"

namespace nn {

class Metric {
    using BatchSig = Scalar(const Matrix&, const Matrix&);

   public:
    Metric();
    explicit Metric(std::function<BatchSig> value);

    Scalar Value(const Matrix& Y_true_cols, const Matrix& Y_logits_cols) const;

    static Metric Accuracy();
    static Metric CrossEntropy();

   private:
    std::function<BatchSig> value_;
};

}  // namespace nn
