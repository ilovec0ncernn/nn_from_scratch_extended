#include "ActivationFunctions.h"

#include <utility>

namespace nn {

static Vector ReluForward(const Vector& z) {
    return z.cwiseMax(Scalar(0));
}
static Vector ReluBackward(const Vector& y, const Vector& dL_dy) {
    return ((y.array() > Scalar(0)).template cast<Scalar>() * dL_dy.array()).matrix();
}

static Vector IdForward(const Vector& z) {
    return z;
}
static Vector IdBackward(const Vector&, const Vector& dL_dy) {
    return dL_dy;
}

static Vector SoftmaxForward(const Vector& z) {
    const Scalar m = z.maxCoeff();
    const Vector e = (z.array() - m).exp().matrix();
    return e / e.sum();
}
static Vector SoftmaxBackward(const Vector& y, const Vector& dL_dy) {
    const Scalar dot = y.dot(dL_dy);
    return (y.array() * (dL_dy.array() - dot)).matrix();
}

Activation::Activation() = default;

Activation::Activation(std::function<ForwardSig> fwd, std::function<BackwardSig> bwd)
    : forward_(std::move(fwd)), backward_(std::move(bwd)) {
}

Vector Activation::Forward(const Vector& z) const {
    return forward_ ? forward_(z) : z;
}
Vector Activation::Backward(const Vector& y, const Vector& dL_dy) const {
    return backward_ ? backward_(y, dL_dy) : dL_dy;
}

Activation Activation::ReLU() {
    return Activation{ReluForward, ReluBackward};
}
Activation Activation::Identity() {
    return Activation{IdForward, IdBackward};
}
Activation Activation::Softmax() {
    return Activation{SoftmaxForward, SoftmaxBackward};
}

}  // namespace nn
