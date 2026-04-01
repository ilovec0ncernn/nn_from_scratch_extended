#include "Except.h"

#include <exception>
#include <iostream>

namespace except {

void React() {
    try {
        throw;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
}

}  // namespace except
