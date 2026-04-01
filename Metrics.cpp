#include "Metrics.h"

#include <cmath>
#include <utility>

namespace nn {

static Scalar AccuracyBatch(const Matrix& Y_true, const Matrix& Y_logits) {
    const Index b = Y_true.cols();

    Index correct = 0;
    for (Index j = 0; j < b; ++j) {
        Index yi = 0;
        Index pi = 0;
        Y_true.col(j).maxCoeff(&yi);
        Y_logits.col(j).maxCoeff(&pi);
        if (yi == pi) {
            ++correct;
        }
    }
    return correct / Scalar(b);
}

static Scalar CrossEntropyBatch(const Matrix& Y_true, const Matrix& logits) {
    const Index b = Y_true.cols();
    Scalar sum = 0;
    for (Index j = 0; j < b; ++j) {
        const Vector z = logits.col(j);
        const Scalar m = z.maxCoeff();
        const Vector exps = (z.array() - m).exp().matrix();
        const Scalar lse = m + std::log(exps.sum());
        sum += -(Y_true.col(j).array() * (z.array() - lse)).sum();
    }
    return sum / Scalar(b);
}

Metric::Metric() = default;

Metric::Metric(std::function<BatchSig> value) : value_(std::move(value)) {
}

Scalar Metric::Value(const Matrix& Y_true_cols, const Matrix& Y_logits_cols) const {
    return value_(Y_true_cols, Y_logits_cols);
}

Metric Metric::Accuracy() {
    return Metric{AccuracyBatch};
}
Metric Metric::CrossEntropy() {
    return Metric{CrossEntropyBatch};
}

}  // namespace nn
