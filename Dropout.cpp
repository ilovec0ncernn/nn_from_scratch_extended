#include "Dropout.h"

#include <random>

namespace nn {

Dropout::Dropout(Scalar drop_rate, std::uint64_t seed)
    : drop_rate_(drop_rate), scale_(Scalar(1) / (Scalar(1) - drop_rate)), rng_(seed) {
}

Matrix Dropout::Forward(const Matrix& X) {
    if (!training_)
        return X;

    std::bernoulli_distribution dist(1.0 - static_cast<double>(drop_rate_));
    mask_.resize(X.rows(), X.cols());
    for (Index c = 0; c < X.cols(); ++c)
        for (Index r = 0; r < X.rows(); ++r)
            mask_(r, c) = dist(rng_) ? scale_ : Scalar(0);
    return X.cwiseProduct(mask_);
}

Matrix Dropout::BackwardDy(const Matrix& dL_dy) {
    if (!training_)
        return dL_dy;
    return dL_dy.cwiseProduct(mask_);
}

Vector Dropout::Forward(const Vector& x) {
    Matrix Xm = x;
    return Forward(Xm).col(0);
}

Vector Dropout::BackwardDy(const Vector& dL_dy) {
    Matrix dY = dL_dy;
    return BackwardDy(dY).col(0);
}

void Dropout::Step(int, Scalar) {
}
void Dropout::SetLr(Scalar) {
}
void Dropout::SetTraining(bool training) {
    training_ = training;
}

void Dropout::ClearCache() {
    mask_.resize(0, 0);
}

void Dropout::SaveWeights(std::ostream&) const {
}
void Dropout::LoadWeights(std::istream&) {
}

}  // namespace nn
