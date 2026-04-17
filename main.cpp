#include <iostream>

#include "Except.h"
#include "Test.h"

int main() {
    try {
        nn::RunAllTests();
    } catch (...) {
        except::React();
    }
    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}
