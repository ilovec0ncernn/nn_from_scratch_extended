#include "Layer.h"

#include <EigenRand/EigenRand>
#include <utility>

namespace nn {

Matrix Layer::InitA(Index out_dim, Index in_dim, RNG& rng) {
    const Scalar std = std::sqrt(Scalar(2) / static_cast<Scalar>(in_dim + out_dim));
    return Eigen::Rand::normal<Matrix>(out_dim, in_dim, rng.gen) * std;
}

Vector Layer::InitB(Index out_dim) {
    return Vector::Constant(out_dim, Scalar(0.01f));
}

Layer::Layer(Index in_dim, Index out_dim, Activation sigma, RNG& rng)
    : A_(InitA(out_dim, in_dim, rng)), b_(InitB(out_dim)), sigma_(std::move(sigma)) {
}

Index Layer::InDim() const {
    return static_cast<Index>(A_.cols());
}
Index Layer::OutDim() const {
    return static_cast<Index>(A_.rows());
}

Vector Layer::Forward(const Vector& x) {
    Matrix Xm = x;
    Matrix Ym = Forward(Xm);
    return Ym.col(0);
}

Vector Layer::BackwardDy(const Vector& dL_dy) {
    Matrix dY = dL_dy;
    Matrix dX = BackwardDy(dY);
    return dX.col(0);
}

Matrix Layer::Forward(const Matrix& X) {
    x_ = X;

    z_.resize(OutDim(), X.cols());
    z_.colwise() = b_;
    z_.noalias() += A_ * X;

    y_.resize(OutDim(), X.cols());
    for (Index c = 0; c < X.cols(); ++c) {
        y_.col(c) = sigma_.Forward(z_.col(c));
    }
    return y_;
}

Matrix Layer::BackwardDy(const Matrix& dL_dy) {
    const Index B = dL_dy.cols();

    Matrix dL_dz(OutDim(), B);
    for (Index c = 0; c < B; ++c) {
        dL_dz.col(c) = sigma_.Backward(y_.col(c), dL_dy.col(c));
    }

    if (dA_sum_.rows() != OutDim() || dA_sum_.cols() != InDim()) {
        dA_sum_.resize(OutDim(), InDim());
    }
    if (db_sum_.size() != OutDim()) {
        db_sum_.resize(OutDim());
    }

    dA_sum_.noalias() = dL_dz * x_.transpose();
    db_sum_.noalias() = dL_dz.rowwise().sum();

    return A_.transpose() * dL_dz;
}

void Layer::Step(float lr, int batch_size) {
    const Scalar scale = lr / static_cast<Scalar>(batch_size);
    A_ -= scale * dA_sum_;
    b_ -= scale * db_sum_;
    if (dA_sum_.size() != 0) {
        dA_sum_.setZero();
    }
    if (db_sum_.size() != 0) {
        db_sum_.setZero();
    }
}

}  // namespace nn
