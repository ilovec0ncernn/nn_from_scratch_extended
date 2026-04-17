#pragma once

#include <cstdint>
#include <iosfwd>
#include <random>

#include "Alias.h"

namespace nn {

class Dropout {
   public:
    explicit Dropout(Scalar drop_rate, std::uint64_t seed = 42);

    Vector Forward(const Vector& x);
    Vector BackwardDy(const Vector& dL_dy);

    Matrix Forward(const Matrix& X);
    Matrix BackwardDy(const Matrix& dL_dy);

    void Step(int batch_size, Scalar lambda = 0.0f);
    void SetLr(Scalar lr);
    void SetTraining(bool training);
    void ClearCache();
    void SaveWeights(std::ostream& out) const;
    void LoadWeights(std::istream& in);

   private:
    Scalar drop_rate_;
    Scalar scale_;
    bool training_ = true;
    std::mt19937_64 rng_;
    Matrix mask_;
};

}  // namespace nn
