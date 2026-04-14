# Neural Networks from Scratch

## Requirements

- **CMake** 4.0+
- **GCC 11+** с поддержкой C++17
- **Git** (для инициализации submodules)

На Windows рекомендуется MinGW.

Все остальные зависимости (Eigen, EigenRand, MNIST, CIFAR-10 reader) по дефолту подключены как git submodules.

## Работа с CIFAR-10

Для корректной работы с датасетом CIFAR-10 необходимо положить папку с бинарниками cifar-10-batches-bin в /external_submodules/CIFAR-10/


## Сборка и запуск проекта

Команда для корректной работы внешних модулей:

```bash
git submodule update --init --recursive
```

Сборка проекта на из корневой папки:

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
.\build\nn_run.exe
```

или

```bash
cmake -S nn_from_scratch_extended -B nn_from_scratch_extended/build -DCMAKE_BUILD_TYPE=Release
cmake --build nn_from_scratch_extended/build -j 8
.\build\nn_run.exe
```