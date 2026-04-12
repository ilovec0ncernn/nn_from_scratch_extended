#pragma once

#include <Eigen/Dense>
#include <iosfwd>

#include "ActivationFunctions.h"
#include "Alias.h"
#include "Optimizer.h"
#include "WeightInit.h"

namespace nn {

class Layer {
   public:
    Layer(Index in_dim, Index out_dim, Activation sigma, RNG& rng, WeightInit init, Optimizer opt);

    Vector Forward(const Vector& x);
    Vector BackwardDy(const Vector& dL_dy);

    Matrix Forward(const Matrix& X);
    Matrix BackwardDy(const Matrix& dL_dy);

    void Step(int batch_size);
    void SetLr(Scalar lr);

    Index InDim() const;
    Index OutDim() const;

    void ClearCache();

    void SaveWeights(std::ostream& out) const;
    void LoadWeights(std::istream& in);

   private:
    static Matrix InitA(Index out_dim, Index in_dim, RNG& rng, WeightInit init);
    static Vector InitB(Index out_dim);

    Matrix A_;
    Vector b_;
    Activation sigma_;

    Matrix x_, z_, y_;
    Matrix dA_sum_;
    Vector db_sum_;

    Optimizer opt_;
    Optimizer::State state_A_;
    Optimizer::State state_b_;
};

}  // namespace nn
