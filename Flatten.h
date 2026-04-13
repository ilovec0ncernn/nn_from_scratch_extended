#pragma once

#include <iosfwd>

#include "Alias.h"

namespace nn {

class Flatten {
   public:
    Flatten() = default;

    Vector Forward(const Vector& x) {
        return x;
    }
    Vector BackwardDy(const Vector& dL_dy) {
        return dL_dy;
    }
    Matrix Forward(const Matrix& X) {
        return X;
    }
    Matrix BackwardDy(const Matrix& dL_dy) {
        return dL_dy;
    }

    void Step(int, Scalar) {
    }
    void SetLr(Scalar) {
    }
    void SetTraining(bool) {
    }
    void ClearCache() {
    }
    void SaveWeights(std::ostream&) const {
    }
    void LoadWeights(std::istream&) {
    }
};

}  // namespace nn
