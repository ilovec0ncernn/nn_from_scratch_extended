#pragma once

#include <functional>
#include <memory>

#include "Alias.h"

namespace nn {

class Optimizer {
   public:
    struct State {
        Matrix m1, m2;
        int step = 0;
    };

    Optimizer() = default;

    void Apply(State& state, Eigen::Ref<Matrix> param, Eigen::Ref<const Matrix> grad, int batch_size) const;
    void SetLr(Scalar lr);
    Scalar Lr() const;

    static Optimizer SGD(Scalar lr, Scalar momentum = 0.0f);
    static Optimizer Adam(Scalar lr, Scalar beta1 = 0.9f, Scalar beta2 = 0.999f, Scalar eps = 1e-8f);

   private:
    using ApplyFn = std::function<void(State&, Eigen::Ref<Matrix>, Eigen::Ref<const Matrix>, int)>;
    ApplyFn apply_fn_;
    std::shared_ptr<Scalar> lr_;
};

}  // namespace nn
