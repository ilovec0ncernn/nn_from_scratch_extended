#include "Metrics.h"

#include <cmath>
#include <utility>
#include <vector>

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

static void ArgmaxCols(const Matrix& Y_true, const Matrix& Y_logits, std::vector<Index>& true_cls,
                       std::vector<Index>& pred_cls) {
    const Index B = Y_true.cols();
    true_cls.resize(B);
    pred_cls.resize(B);
    for (Index j = 0; j < B; ++j) {
        Y_true.col(j).maxCoeff(&true_cls[j]);
        Y_logits.col(j).maxCoeff(&pred_cls[j]);
    }
}

static Scalar PrecisionBatch(const Matrix& Y_true, const Matrix& Y_logits) {
    const Index C = Y_true.rows();
    const Index B = Y_true.cols();
    std::vector<Index> true_cls, pred_cls;
    ArgmaxCols(Y_true, Y_logits, true_cls, pred_cls);

    Scalar sum = Scalar(0);
    for (Index k = 0; k < C; ++k) {
        int tp = 0, fp = 0;
        for (Index j = 0; j < B; ++j) {
            if (pred_cls[j] == k) {
                if (true_cls[j] == k)
                    ++tp;
                else
                    ++fp;
            }
        }
        if (tp + fp > 0)
            sum += Scalar(tp) / Scalar(tp + fp);
    }
    return sum / Scalar(C);
}

static Scalar RecallBatch(const Matrix& Y_true, const Matrix& Y_logits) {
    const Index C = Y_true.rows();
    const Index B = Y_true.cols();
    std::vector<Index> true_cls, pred_cls;
    ArgmaxCols(Y_true, Y_logits, true_cls, pred_cls);

    Scalar sum = Scalar(0);
    for (Index k = 0; k < C; ++k) {
        int tp = 0, fn = 0;
        for (Index j = 0; j < B; ++j) {
            if (true_cls[j] == k) {
                if (pred_cls[j] == k)
                    ++tp;
                else
                    ++fn;
            }
        }
        if (tp + fn > 0)
            sum += Scalar(tp) / Scalar(tp + fn);
    }
    return sum / Scalar(C);
}

Metric Metric::Accuracy() {
    return Metric{AccuracyBatch};
}
Metric Metric::CrossEntropy() {
    return Metric{CrossEntropyBatch};
}
Metric Metric::Precision() {
    return Metric{PrecisionBatch};
}
Metric Metric::Recall() {
    return Metric{RecallBatch};
}

}  // namespace nn
