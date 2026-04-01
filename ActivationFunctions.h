#pragma once

#include <functional>

#include "Alias.h"

namespace nn {

class Activation {
    using ForwardSig = Vector(const Vector&);
    using BackwardSig = Vector(const Vector&, const Vector&);

   public:
    Activation();
    Activation(std::function<ForwardSig> fwd, std::function<BackwardSig> bwd);

    Vector Forward(const Vector& z) const;
    Vector Backward(const Vector& y, const Vector& dL_dy) const;

    static Activation ReLU();
    static Activation Identity();
    static Activation Softmax();

   private:
    std::function<ForwardSig> forward_;
    std::function<BackwardSig> backward_;
};

}  // namespace nn
