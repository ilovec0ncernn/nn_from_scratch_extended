#include "InputDataset.h"

#ifdef _MSC_VER
#include "mnist/mnist_reader_less.hpp"
#else
#include "mnist/mnist_reader.hpp"
#endif

#include "cifar/cifar10_reader.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nn {

static Vector ToOneHot(Index label) {
    Vector y = Vector::Zero(10);
    y[label] = 1.0f;
    return y;
}

static Scalar Normalize(uint8_t pixel) noexcept {
    return static_cast<Scalar>(pixel) / 255.0f;
}

static std::pair<Matrix, Matrix> MakeXY(const std::vector<std::vector<uint8_t>>& images,
                                        const std::vector<uint8_t>& labels) {
    const Index n = static_cast<Index>(images.size());

    Matrix X(784, n);
    Matrix y(10, n);

    for (Index i = 0; i < n; ++i) {
        const auto& img = images[static_cast<size_t>(i)];
        for (Index j = 0; j < 784; ++j) {
            X(j, i) = Normalize(img[static_cast<size_t>(j)]);
        }
        const Index lab = static_cast<Index>(labels[static_cast<size_t>(i)]);
        y.col(i) = ToOneHot(lab);
    }
    return {std::move(X), std::move(y)};
}

static std::pair<Matrix, Matrix> MakeCifarXY(const std::vector<std::vector<uint8_t>>& images,
                                              const std::vector<uint8_t>& labels) {
    const Index n = static_cast<Index>(images.size());

    Matrix X(3072, n);
    Matrix y(10, n);

    for (Index i = 0; i < n; ++i) {
        const auto& img = images[static_cast<size_t>(i)];
        for (Index j = 0; j < 3072; ++j) {
            X(j, i) = Normalize(img[static_cast<size_t>(j)]);
        }
        const Index lab = static_cast<Index>(labels[static_cast<size_t>(i)]);
        y.col(i) = ToOneHot(lab);
    }
    return {std::move(X), std::move(y)};
}

Split InputDataset::LoadMnist(const std::filesystem::path& dir) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(dir.string());
    if (dataset.training_images.empty() || dataset.test_images.empty()) {
        throw std::runtime_error(std::string("mnist files not found in \"") + dir.string() + "\"");
    }

    Split split;
    {
        auto xy = MakeXY(dataset.training_images, dataset.training_labels);
        split.X_train = std::move(xy.first);
        split.y_train = std::move(xy.second);
    }
    {
        auto xy = MakeXY(dataset.test_images, dataset.test_labels);
        split.X_test = std::move(xy.first);
        split.y_test = std::move(xy.second);
    }
    return split;
}

Split InputDataset::LoadCifar10(const std::filesystem::path& dir) {
    cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
    cifar::read_training(dir.string(), 0, dataset.training_images, dataset.training_labels,
                         [] { return std::vector<uint8_t>(3072); });
    cifar::read_test(dir.string(), 0, dataset.test_images, dataset.test_labels,
                     [] { return std::vector<uint8_t>(3072); });

    if (dataset.training_images.empty() || dataset.test_images.empty()) {
        throw std::runtime_error(std::string("CIFAR-10 files not found in \"") + dir.string() + "\"");
    }

    Split split;
    {
        auto xy = MakeCifarXY(dataset.training_images, dataset.training_labels);
        split.X_train = std::move(xy.first);
        split.y_train = std::move(xy.second);
    }
    {
        auto xy = MakeCifarXY(dataset.test_images, dataset.test_labels);
        split.X_test = std::move(xy.first);
        split.y_test = std::move(xy.second);
    }
    return split;
}

}  // namespace nn
