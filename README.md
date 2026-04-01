# Neural Networks from Scratch (C++17)

## Зависимости aka requirements

- Система сборки: CMake 4.0
- Система поддержки версий: Git 2.34
- Компилятор C++17 (MingW)
- Windows PowerShell


## Сборка и запуск проекта

Команда для корректной работы внешних модулей:

```bash
git submodule update --init --recursive
```

Сборка проекта на Windows (MingW) из корневой папки:

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
.\build\nn_run.exe
```