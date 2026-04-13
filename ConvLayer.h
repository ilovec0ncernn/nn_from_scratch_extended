#pragma once

#include <Eigen/Dense>
#include <iosfwd>

#include "ActivationFunctions.h"
#include "Alias.h"
#include "Optimizer.h"
#include "WeightInit.h"

namespace nn {

class ConvLayer {
   public:
    ConvLayer(Index C_in, Index H_in, Index W_in, Index C_out, Index kH, Index kW, RNG& rng, Activation sigma,
              WeightInit init, Optimizer opt, Index stride = 1, Index pad = 0);

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

    Index OutChannels() const;
    Index OutH() const;
    Index OutW() const;

   private:
    static Matrix InitW(Index C_out, Index C_in, Index kH, Index kW, RNG& rng, WeightInit init);
    static Vector InitB(Index C_out);

    Matrix Im2Col(const Matrix& X) const;
    Matrix Col2Im(const Matrix& dcol, Index B) const;

    Index C_in_, H_in_, W_in_;
    Index C_out_, kH_, kW_;
    Index stride_, pad_;
    Index H_out_, W_out_;

    Matrix W_;
    Vector b_;
    Activation sigma_;

    Matrix col_;
    Matrix y_;

    Matrix dW_sum_;
    Vector db_sum_;

    Optimizer opt_;
    Optimizer::State state_W_;
    Optimizer::State state_b_;
};

}  // namespace nn
