#pragma once

#include <iosfwd>

#include "Alias.h"

namespace nn {

class MaxPool {
   public:
    MaxPool(Index C, Index H_in, Index W_in, Index kH, Index kW, Index stride);

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
    Index C_, H_in_, W_in_;
    Index kH_, kW_, stride_;
    Index H_out_, W_out_;
    Matrix mask_;
};

}  // namespace nn
