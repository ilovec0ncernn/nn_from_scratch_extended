#pragma once

#include <cstdint>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace nn {

using Scalar = float;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Index = Eigen::Index;

struct RNG {
    Eigen::Rand::P8_mt19937_64 gen{42};
};

}  // namespace nn
