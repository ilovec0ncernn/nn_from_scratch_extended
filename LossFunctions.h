#pragma once

#include <functional>
#include "Alias.h"

namespace nn {

class Loss {
    using LossSignature = Scalar(const Vector&, const Vector&);
    using GradSignature = Vector(const Vector&, const Vector&);

   public:
    Loss();
    Loss(std::function<LossSignature> loss, std::function<GradSignature> grad);

    Scalar LossVal(const Vector& y_true, const Vector& y_pred_or_logits) const;
    Vector Gradient(const Vector& y_true, const Vector& y_pred_or_logits) const;

    Scalar LossVal(const Matrix& Y_true, const Matrix& Y_pred_or_logits) const;
    Matrix Gradient(const Matrix& Y_true, const Matrix& Y_pred_or_logits) const;

    static Loss Mse();
    static Loss CrossEntropy();

   private:
    std::function<LossSignature> loss_;
    std::function<GradSignature> grad_;
};

}  // namespace nn
