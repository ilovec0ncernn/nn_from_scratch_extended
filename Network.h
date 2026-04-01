#pragma once

#include <random>
#include <vector>

#include "Alias.h"
#include "Layer.h"
#include "LossFunctions.h"
#include "Metrics.h"

namespace nn {

struct TrainConfig {
    int epochs = 10;
    int batch_size = 64;
    float lr = 0.05f;
    std::uint64_t shuffle_seed = 42;
};

class Network {
   public:
    Network() = default;

    Network& AddFirstLayer(Index in_dim, Index out_dim, Activation sigma, RNG& rng);
    Network& AddLayer(Index out_dim, Activation sigma, RNG& rng);

    void Train(const Matrix& X_cols, const Matrix& Y_cols, const Matrix& X_val_cols, const Matrix& Y_val_cols,
               const TrainConfig& cfg, const Loss& loss);

    Matrix Predict(const Matrix& X_cols);
    Vector PredictOne(const Vector& x);

   private:
    Matrix ForwardAll(const Matrix& Xb);
    Matrix BackwardAll(const Matrix& dY);
    void StepAll(Scalar lr, int batch_size);

    std::vector<Layer> layers_;
    Index last_dim_ = -1;
    bool has_input_dim_ = false;
};

}  // namespace nn
