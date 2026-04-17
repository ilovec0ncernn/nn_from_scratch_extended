#include "Optimizer.h"

#include <cmath>

namespace nn {

void Optimizer::Apply(State& s, Eigen::Ref<Matrix> param, Eigen::Ref<const Matrix> grad, int bs) const {
    apply_fn_(s, param, grad, bs);
}

void Optimizer::SetLr(Scalar lr) {
    *lr_ = lr;
}

Scalar Optimizer::Lr() const {
    return *lr_;
}

Optimizer Optimizer::SGD(Scalar lr, Scalar momentum) {
    Optimizer opt;
    opt.lr_ = std::make_shared<Scalar>(lr);
    auto lr_ptr = opt.lr_;
    opt.apply_fn_ = [lr_ptr, momentum](State& s, Eigen::Ref<Matrix> param, Eigen::Ref<const Matrix> grad, int bs) {
        const Scalar eta = *lr_ptr / Scalar(bs);
        if (s.m1.size() == 0)
            s.m1 = Matrix::Zero(param.rows(), param.cols());
        s.m1 = momentum * s.m1 - eta * grad;
        param += s.m1;
    };
    return opt;
}

Optimizer Optimizer::Adam(Scalar lr, Scalar beta1, Scalar beta2, Scalar eps) {
    Optimizer opt;
    opt.lr_ = std::make_shared<Scalar>(lr);
    auto lr_ptr = opt.lr_;
    opt.apply_fn_ = [lr_ptr, beta1, beta2, eps](State& s, Eigen::Ref<Matrix> param, Eigen::Ref<const Matrix> grad,
                                                int bs) {
        const Matrix g = grad / Scalar(bs);
        ++s.step;
        if (s.m1.size() == 0) {
            s.m1 = Matrix::Zero(param.rows(), param.cols());
            s.m2 = Matrix::Zero(param.rows(), param.cols());
        }
        s.m1 = beta1 * s.m1 + (Scalar(1) - beta1) * g;
        s.m2 = beta2 * s.m2 + (Scalar(1) - beta2) * g.cwiseProduct(g);
        const Scalar bc1 = Scalar(1) - std::pow(beta1, Scalar(s.step));
        const Scalar bc2 = Scalar(1) - std::pow(beta2, Scalar(s.step));
        const Matrix m_hat = s.m1 / bc1;
        const Matrix v_hat = s.m2 / bc2;
        param.array() -= *lr_ptr * m_hat.array() / (v_hat.array().sqrt() + eps);
    };
    return opt;
}

}  // namespace nn
