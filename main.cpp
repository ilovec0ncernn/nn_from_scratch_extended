#include "Except.h"
#include "Test.h"

int main() {
    try {
        nn::RunAllTests();
    } catch (...) {
        except::React();
    }
    return 0;
}
