#include "ConvLayer.h"

#include <EigenRand/EigenRand>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace nn {

Matrix ConvLayer::InitW(Index C_out, Index C_in, Index kH, Index kW, RNG& rng, WeightInit init) {
    const Index fan_in = C_in * kH * kW;
    const Index fan_out = C_out * kH * kW;
    Scalar std;
    switch (init) {
        case WeightInit::He:
            std = std::sqrt(Scalar(2) / static_cast<Scalar>(fan_in));
            break;
        case WeightInit::Xavier:
        default:
            std = std::sqrt(Scalar(2) / static_cast<Scalar>(fan_in + fan_out));
            break;
    }
    return Eigen::Rand::normal<Matrix>(C_out, C_in * kH * kW, rng.gen) * std;
}

Vector ConvLayer::InitB(Index C_out) {
    return Vector::Constant(C_out, Scalar(0.01f));
}

ConvLayer::ConvLayer(Index C_in, Index H_in, Index W_in, Index C_out, Index kH, Index kW, RNG& rng, Activation sigma,
                     WeightInit init, Optimizer opt, Index stride, Index pad)
    : C_in_(C_in)
    , H_in_(H_in)
    , W_in_(W_in)
    , C_out_(C_out)
    , kH_(kH)
    , kW_(kW)
    , stride_(stride)
    , pad_(pad)
    , H_out_((H_in + 2 * pad - kH) / stride + 1)
    , W_out_((W_in + 2 * pad - kW) / stride + 1)
    , W_(InitW(C_out, C_in, kH, kW, rng, init))
    , b_(InitB(C_out))
    , sigma_(std::move(sigma))
    , opt_(std::move(opt)) {
}

Index ConvLayer::OutChannels() const {
    return C_out_;
}
Index ConvLayer::OutH() const {
    return H_out_;
}
Index ConvLayer::OutW() const {
    return W_out_;
}

Matrix ConvLayer::Im2Col(const Matrix& X) const {
    const Index B = X.cols();
    const Index col_rows = C_in_ * kH_ * kW_;
    const Index col_cols = H_out_ * W_out_ * B;
    Matrix col = Matrix::Zero(col_rows, col_cols);
    for (Index b = 0; b < B; ++b) {
        for (Index oh = 0; oh < H_out_; ++oh) {
            for (Index ow = 0; ow < W_out_; ++ow) {
                const Index col_c = b * H_out_ * W_out_ + oh * W_out_ + ow;
                for (Index c = 0; c < C_in_; ++c) {
                    for (Index kh = 0; kh < kH_; ++kh) {
                        for (Index kw = 0; kw < kW_; ++kw) {
                            const Index ih = oh * stride_ - pad_ + kh;
                            const Index iw = ow * stride_ - pad_ + kw;
                            if (ih >= 0 && ih < H_in_ && iw >= 0 && iw < W_in_) {
                                const Index row = c * kH_ * kW_ + kh * kW_ + kw;
                                col(row, col_c) = X(c * H_in_ * W_in_ + ih * W_in_ + iw, b);
                            }
                        }
                    }
                }
            }
        }
    }
    return col;
}

Matrix ConvLayer::Col2Im(const Matrix& dcol, Index B) const {
    Matrix dX = Matrix::Zero(C_in_ * H_in_ * W_in_, B);
    for (Index b = 0; b < B; ++b) {
        for (Index oh = 0; oh < H_out_; ++oh) {
            for (Index ow = 0; ow < W_out_; ++ow) {
                const Index col_c = b * H_out_ * W_out_ + oh * W_out_ + ow;
                for (Index c = 0; c < C_in_; ++c) {
                    for (Index kh = 0; kh < kH_; ++kh) {
                        for (Index kw = 0; kw < kW_; ++kw) {
                            const Index ih = oh * stride_ - pad_ + kh;
                            const Index iw = ow * stride_ - pad_ + kw;
                            if (ih >= 0 && ih < H_in_ && iw >= 0 && iw < W_in_) {
                                const Index row = c * kH_ * kW_ + kh * kW_ + kw;
                                dX(c * H_in_ * W_in_ + ih * W_in_ + iw, b) += dcol(row, col_c);
                            }
                        }
                    }
                }
            }
        }
    }
    return dX;
}

Matrix ConvLayer::Forward(const Matrix& X) {
    const Index B = X.cols();
    const Index Hs = H_out_ * W_out_;

    col_ = Im2Col(X);

    Matrix Z = W_ * col_;
    for (Index i = 0; i < Hs * B; ++i) {
        Z.col(i) += b_;
    }

    y_.resize(C_out_ * Hs, B);
    Vector z_col(C_out_ * Hs);
    for (Index b = 0; b < B; ++b) {
        for (Index c = 0; c < C_out_; ++c) {
            for (Index hs = 0; hs < Hs; ++hs) {
                z_col(c * Hs + hs) = Z(c, b * Hs + hs);
            }
        }
        y_.col(b) = sigma_.Forward(z_col);
    }

    return y_;
}

Matrix ConvLayer::BackwardDy(const Matrix& dL_dy) {
    const Index B = dL_dy.cols();
    const Index Hs = H_out_ * W_out_;

    Matrix dL_dz(C_out_ * Hs, B);
    for (Index b = 0; b < B; ++b)
        dL_dz.col(b) = sigma_.Backward(y_.col(b), dL_dy.col(b));

    Matrix dL_dZ(C_out_, Hs * B);
    for (Index b = 0; b < B; ++b) {
        for (Index c = 0; c < C_out_; ++c) {
            for (Index hs = 0; hs < Hs; ++hs) {
                dL_dZ(c, b * Hs + hs) = dL_dz(c * Hs + hs, b);
            }
        }
    }

    if (dW_sum_.rows() != C_out_ || dW_sum_.cols() != C_in_ * kH_ * kW_)
        dW_sum_.resize(C_out_, C_in_ * kH_ * kW_);
    if (db_sum_.size() != C_out_)
        db_sum_.resize(C_out_);

    dW_sum_.noalias() = dL_dZ * col_.transpose();
    db_sum_.noalias() = dL_dZ.rowwise().sum();

    Matrix dL_dcol = W_.transpose() * dL_dZ;
    return Col2Im(dL_dcol, B);
}

Vector ConvLayer::Forward(const Vector& x) {
    Matrix Xm = x;
    return Forward(Xm).col(0);
}

Vector ConvLayer::BackwardDy(const Vector& dL_dy) {
    Matrix dY = dL_dy;
    return BackwardDy(dY).col(0);
}

void ConvLayer::Step(int batch_size, Scalar lambda) {
    if (lambda != Scalar(0)) {
        dW_sum_.noalias() += Scalar(batch_size) * lambda * W_;
    }
    opt_.Apply(state_W_, W_, dW_sum_, batch_size);

    Eigen::Map<Matrix> b_map(b_.data(), b_.rows(), 1);
    Eigen::Map<const Matrix> db_map(db_sum_.data(), db_sum_.rows(), 1);
    opt_.Apply(state_b_, b_map, db_map, batch_size);

    if (dW_sum_.size() != 0) {
        dW_sum_.setZero();
    }
    if (db_sum_.size() != 0) {
        db_sum_.setZero();
    }
}

void ConvLayer::SetLr(Scalar lr) {
    opt_.SetLr(lr);
}

void ConvLayer::SetTraining(bool) {
}

void ConvLayer::ClearCache() {
    col_.resize(0, 0);
    y_.resize(0, 0);
    dW_sum_.resize(0, 0);
    db_sum_.resize(0);
}

void ConvLayer::SaveWeights(std::ostream& out) const {
    const int64_t rows = W_.rows();
    const int64_t cols = W_.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(W_.data()), rows * cols * sizeof(Scalar));

    const int64_t sz = b_.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    out.write(reinterpret_cast<const char*>(b_.data()), sz * sizeof(Scalar));
}

void ConvLayer::LoadWeights(std::istream& in) {
    int64_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (rows != W_.rows() || cols != W_.cols()) {
        throw std::runtime_error("ConvLayer::LoadWeights: weight matrix dimension mismatch");
    }
    in.read(reinterpret_cast<char*>(W_.data()), rows * cols * sizeof(Scalar));

    int64_t sz;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    if (sz != b_.size()) {
        throw std::runtime_error("ConvLayer::LoadWeights: bias vector dimension mismatch");
    }
    in.read(reinterpret_cast<char*>(b_.data()), sz * sizeof(Scalar));
}

}  // namespace nn
