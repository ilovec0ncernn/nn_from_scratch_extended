#include "LossFunctions.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace nn {

static Scalar MseLossVec(const Vector& y_true, const Vector& y_pred) {
    const Vector diff = y_pred - y_true;
    return diff.squaredNorm() / static_cast<Scalar>(diff.size());
}

static Vector MseGradVec(const Vector& y_true, const Vector& y_pred) {
    const Scalar invN = Scalar(1) / static_cast<Scalar>(y_true.size());
    return Scalar(2) * invN * (y_pred - y_true);
}

static Scalar CrossEntropyLossVecLogits(const Vector& y_true, const Vector& logits) {
    const Scalar m = logits.maxCoeff();
    const Vector exps = (logits.array() - m).exp().matrix();
    const Scalar lse = m + std::log(exps.sum());
    const Vector log_probs = logits.array() - lse;
    return -(y_true.array() * log_probs.array()).sum();
}

static Vector CrossEntropyGradVecLogits(const Vector& y_true, const Vector& logits) {
    const Scalar m = logits.maxCoeff();
    const Vector exps = (logits.array() - m).exp().matrix();
    const Scalar denom = exps.sum();
    const Vector probs = exps / denom;
    return probs - y_true;
}

Loss::Loss() = default;

Loss::Loss(std::function<LossSignature> loss, std::function<GradSignature> grad)
    : loss_(std::move(loss)), grad_(std::move(grad)) {
}

Scalar Loss::LossVal(const Vector& y_true, const Vector& y_pred_or_logits) const {
    return loss_(y_true, y_pred_or_logits);
}

Vector Loss::Gradient(const Vector& y_true, const Vector& y_pred_or_logits) const {
    return grad_(y_true, y_pred_or_logits);
}

Scalar Loss::LossVal(const Matrix& Y_true, const Matrix& Y_pred_or_logits) const {
    const Index b = Y_true.cols();
    if (b == 0)
        return Scalar(0);
    Scalar sum = 0;
    for (Index j = 0; j < b; ++j) {
        sum += loss_(Y_true.col(j), Y_pred_or_logits.col(j));
    }
    return sum / static_cast<Scalar>(b);
}

Matrix Loss::Gradient(const Matrix& Y_true, const Matrix& Y_pred_or_logits) const {
    const Index b = Y_true.cols();
    if (b == 0)
        return Matrix::Zero(Y_true.rows(), 0);

    Matrix G(Y_true.rows(), b);
    for (Index j = 0; j < b; ++j) {
        G.col(j) = grad_(Y_true.col(j), Y_pred_or_logits.col(j));
    }
    return G;
}

Loss Loss::Mse() {
    return Loss{MseLossVec, MseGradVec};
}

Loss Loss::CrossEntropy() {
    return Loss{CrossEntropyLossVecLogits, CrossEntropyGradVecLogits};
}

}  // namespace nn
