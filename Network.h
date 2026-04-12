#pragma once

#include <filesystem>
#include <random>
#include <vector>

#include "Alias.h"
#include "Layer.h"
#include "LossFunctions.h"
#include "LRScheduler.h"
#include "Metrics.h"
#include "Optimizer.h"
#include "TrainHistory.h"
#include "WeightInit.h"

namespace nn {

struct TrainConfig {
    int epochs = 10;
    int batch_size = 64;
    float lr = 0.05f;
    std::uint64_t shuffle_seed = 42;
    LRScheduler scheduler = LRScheduler::Constant(0.05f);
};

class Network {
   public:
    Network() = default;

    Network& AddFirstLayer(Index in_dim, Index out_dim, Activation sigma, RNG& rng,
                           WeightInit init = WeightInit::Xavier, Optimizer opt = Optimizer::SGD(0.05f));
    Network& AddLayer(Index out_dim, Activation sigma, RNG& rng, WeightInit init = WeightInit::Xavier,
                      Optimizer opt = Optimizer::SGD(0.05f));

    TrainHistory Train(const Matrix& X_cols, const Matrix& Y_cols, const Matrix& X_val_cols, const Matrix& Y_val_cols,
                       const TrainConfig& cfg, const Loss& loss);

    Matrix Predict(const Matrix& X_cols);
    Vector PredictOne(const Vector& x);

    void ClearCache();

    void Save(const std::filesystem::path& path) const;
    void Load(const std::filesystem::path& path);

   private:
    Matrix ForwardAll(const Matrix& Xb);
    Matrix BackwardAll(const Matrix& dY);
    void StepAll(int batch_size);
    void SetLrAll(Scalar lr);

    std::vector<Layer> layers_;
    Index last_dim_ = -1;
    bool has_input_dim_ = false;
};

}  // namespace nn
