#include "Network.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace nn {

Network& Network::AddFirstLayer(Index in_dim, Index out_dim, Activation sigma, RNG& rng, WeightInit init,
                                Optimizer opt) {
    assert(!has_input_dim_ && "AddFirstLayer called twice");
    layers_.push_back(Layer(in_dim, out_dim, std::move(sigma), rng, init, std::move(opt)));
    first_dim_ = in_dim;
    last_dim_ = out_dim;
    has_input_dim_ = true;
    return *this;
}

Network& Network::AddLayer(Index out_dim, Activation sigma, RNG& rng, WeightInit init, Optimizer opt) {
    assert(has_input_dim_ && "call AddFirstLayer() first");
    const Index in_dim = last_dim_;
    layers_.push_back(Layer(in_dim, out_dim, std::move(sigma), rng, init, std::move(opt)));
    last_dim_ = out_dim;
    return *this;
}

Network& Network::AddDropout(Scalar drop_rate, std::uint64_t seed) {
    assert(has_input_dim_ && "call AddFirstLayer() first");
    layers_.push_back(Dropout(drop_rate, seed));
    return *this;
}

Matrix Network::ForwardAll(const Matrix& Xb) {
    Matrix h = Xb;
    for (auto& L : layers_)
        h = std::visit([&h](auto& l) { return l.Forward(h); }, L);
    return h;
}

Matrix Network::BackwardAll(const Matrix& dY) {
    Matrix grad = std::visit([&dY](auto& l) { return l.BackwardDy(dY); }, layers_.back());
    for (auto it = layers_.rbegin() + 1; it != layers_.rend(); ++it)
        grad = std::visit([&grad](auto& l) { return l.BackwardDy(grad); }, *it);
    return grad;
}

void Network::StepAll(int batch_size, Scalar lambda) {
    for (auto& L : layers_)
        std::visit([batch_size, lambda](auto& l) { l.Step(batch_size, lambda); }, L);
}

void Network::SetLrAll(Scalar lr) {
    for (auto& L : layers_)
        std::visit([lr](auto& l) { l.SetLr(lr); }, L);
}

void Network::SetTrainingAll(bool training) {
    for (auto& L : layers_)
        std::visit([training](auto& l) { l.SetTraining(training); }, L);
}

Matrix Network::Predict(const Matrix& X_cols) {
    if (layers_.empty())
        return Matrix::Zero(0, X_cols.cols());
    SetTrainingAll(false);
    return ForwardAll(X_cols);
}

Vector Network::PredictOne(const Vector& x) {
    Matrix X(x.size(), 1);
    X.col(0) = x;
    Matrix Y = Predict(X);
    return Y.col(0);
}

TrainHistory Network::Train(const Matrix& X_cols, const Matrix& Y_cols, const Matrix& X_val_cols,
                            const Matrix& Y_val_cols, const TrainConfig& cfg, const Loss& loss) {
    TrainHistory history;

    const Index n = X_cols.cols();
    if (n == 0) {
        std::cout << "empty training set\n";
        return history;
    }

    const Index din = first_dim_;
    const Index dout = last_dim_;
    const int b = std::max(1, cfg.batch_size);

    std::vector<Index> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::mt19937_64 eng(cfg.shuffle_seed);

    Metric acc = Metric::Accuracy();
    Metric ce = Metric::CrossEntropy();

    history.train_acc.reserve(cfg.epochs);
    history.val_acc.reserve(cfg.epochs);
    history.val_ce.reserve(cfg.epochs);

    SetTrainingAll(true);

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        const Scalar epoch_lr = cfg.scheduler.Step(epoch);
        SetLrAll(epoch_lr);

        std::shuffle(order.begin(), order.end(), eng);
        Scalar sum_acc = Scalar(0);
        Index seen = 0;

        for (Index start = 0; start < n; start += b) {
            const Index remain = n - start;
            const int r = remain < Index(b) ? int(remain) : b;

            Matrix Xb(din, r);
            Matrix Yb(dout, r);
            for (int j = 0; j < r; ++j) {
                const Index idx = order[start + j];
                Xb.col(j) = X_cols.col(idx);
                Yb.col(j) = Y_cols.col(idx);
            }

            Matrix logits = ForwardAll(Xb);
            const Scalar acc_batch = acc.Value(Yb, logits);
            sum_acc += acc_batch * Scalar(r);
            seen += Index(r);

            Matrix dY = loss.Gradient(Yb, logits);
            BackwardAll(dY);
            StepAll(r, cfg.weight_decay);
        }

        const Scalar epoch_train_acc = (seen > 0) ? (sum_acc / Scalar(seen)) : Scalar(0);

        SetTrainingAll(false);
        Matrix logits_val = ForwardAll(X_val_cols);
        SetTrainingAll(true);

        const Scalar epoch_val_acc = acc.Value(Y_val_cols, logits_val);
        const Scalar epoch_val_ce = ce.Value(Y_val_cols, logits_val);

        history.train_acc.push_back(epoch_train_acc);
        history.val_acc.push_back(epoch_val_acc);
        history.val_ce.push_back(epoch_val_ce);

        std::cout << "epoch " << epoch << " (lr=" << std::fixed << std::setprecision(5) << epoch_lr
                  << "): train_acc=" << std::fixed << std::setprecision(4) << epoch_train_acc
                  << ", val_acc=" << std::fixed << std::setprecision(4) << epoch_val_acc << ", val_ce=" << std::fixed
                  << std::setprecision(4) << epoch_val_ce << std::endl;
    }

    ClearCache();
    SetTrainingAll(false);
    return history;
}

void Network::ClearCache() {
    for (auto& L : layers_)
        std::visit([](auto& l) { l.ClearCache(); }, L);
}

void Network::Save(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Network::Save: cannot open file: " + path.string());

    const uint32_t magic = 0x4E4E574F;
    const uint32_t version = 1;
    const uint32_t num_layers = static_cast<uint32_t>(layers_.size());
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    for (const auto& L : layers_)
        std::visit([&out](const auto& l) { l.SaveWeights(out); }, L);

    if (!out)
        throw std::runtime_error("Network::Save: write error");
}

void Network::Load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Network::Load: cannot open file: " + path.string());

    uint32_t magic, version, num_layers;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    if (magic != 0x4E4E574F)
        throw std::runtime_error("Network::Load: invalid file format");
    if (version != 1)
        throw std::runtime_error("Network::Load: unsupported version " + std::to_string(version));
    if (num_layers != static_cast<uint32_t>(layers_.size()))
        throw std::runtime_error("Network::Load: layer count mismatch (file has " + std::to_string(num_layers) +
                                 ", network has " + std::to_string(layers_.size()) + ")");

    for (auto& L : layers_)
        std::visit([&in](auto& l) { l.LoadWeights(in); }, L);

    if (!in)
        throw std::runtime_error("Network::Load: read error or unexpected EOF");
}

}  // namespace nn
