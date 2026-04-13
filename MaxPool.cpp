#include "MaxPool.h"

#include <limits>

namespace nn {

MaxPool::MaxPool(Index C, Index H_in, Index W_in, Index kH, Index kW, Index stride)
    : C_(C)
    , H_in_(H_in)
    , W_in_(W_in)
    , kH_(kH)
    , kW_(kW)
    , stride_(stride)
    , H_out_((H_in - kH) / stride + 1)
    , W_out_((W_in - kW) / stride + 1) {
}

Matrix MaxPool::Forward(const Matrix& X) {
    const Index B = X.cols();
    const Index out_size = C_ * H_out_ * W_out_;

    Matrix Y(out_size, B);
    mask_.resize(out_size, B);

    for (Index b = 0; b < B; ++b) {
        for (Index c = 0; c < C_; ++c) {
            for (Index oh = 0; oh < H_out_; ++oh) {
                for (Index ow = 0; ow < W_out_; ++ow) {
                    Scalar max_val = -std::numeric_limits<Scalar>::infinity();
                    Index max_idx = 0;
                    for (Index kh = 0; kh < kH_; ++kh) {
                        for (Index kw = 0; kw < kW_; ++kw) {
                            const Index ih = oh * stride_ + kh;
                            const Index iw = ow * stride_ + kw;
                            const Index in_idx = c * H_in_ * W_in_ + ih * W_in_ + iw;
                            const Scalar val = X(in_idx, b);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = in_idx;
                            }
                        }
                    }
                    const Index out_idx = c * H_out_ * W_out_ + oh * W_out_ + ow;
                    Y(out_idx, b) = max_val;
                    mask_(out_idx, b) = static_cast<Scalar>(max_idx);
                }
            }
        }
    }
    return Y;
}

Matrix MaxPool::BackwardDy(const Matrix& dL_dy) {
    const Index B = dL_dy.cols();
    Matrix dX = Matrix::Zero(C_ * H_in_ * W_in_, B);
    const Index out_size = C_ * H_out_ * W_out_;
    for (Index b = 0; b < B; ++b) {
        for (Index out_idx = 0; out_idx < out_size; ++out_idx) {
            const Index in_idx = static_cast<Index>(mask_(out_idx, b));
            dX(in_idx, b) += dL_dy(out_idx, b);
        }
    }
    return dX;
}

Vector MaxPool::Forward(const Vector& x) {
    Matrix Xm = x;
    return Forward(Xm).col(0);
}

Vector MaxPool::BackwardDy(const Vector& dL_dy) {
    Matrix dY = dL_dy;
    return BackwardDy(dY).col(0);
}

void MaxPool::Step(int, Scalar) {
}
void MaxPool::SetLr(Scalar) {
}
void MaxPool::SetTraining(bool) {
}

void MaxPool::ClearCache() {
    mask_.resize(0, 0);
}

void MaxPool::SaveWeights(std::ostream&) const {
}
void MaxPool::LoadWeights(std::istream&) {
}

}  // namespace nn
