// CPU-only bounds and offset tests for the local GPU capture address parser.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <cstring>
#include "nvprof.h"

int main()
{
    unsigned long long values[] = {0x1000, 0x8000, 0, ~0ULL - 3};
    prof::CuLaunch launch = {};
    launch.params = values;
    launch.param_size = sizeof(values);
    struct Case { const char *spec; bool valid; unsigned long long expected; };
    const Case cases[] = {
        {"p0", true, 0x1000}, {"p1+1966080", true, 0x8000 + 1966080},
        {"p0+0x20", true, 0x1020}, {"p0+0", true, 0x1000}, {"0x1234", true, 0x1234},
        {"4096", true, 4096}, {"p3+3", true, ~0ULL}, {"p3+4", false, 0},
        {"p4", false, 0}, {"p4294967295", false, 0}, {"p18446744073709551616", false, 0},
        {"p", false, 0}, {"p-1", false, 0}, {"p0-1", false, 0}, {"p0+", false, 0},
        {"p0+-1", false, 0}, {"p0++1", false, 0}, {"p0+18446744073709551616", false, 0},
        {"p0+1junk", false, 0}, {"p0junk", false, 0}, {"", false, 0},
        {"-1", false, 0}, {" 123", false, 0}, {"0x123junk", false, 0}, {"p2", false, 0}, {"0", false, 0}
    };
    unsigned count = 0;
    for (const Case &test : cases) {
        unsigned long long output = 0x42;
        const bool valid = prof::ResolveTapAddress(test.spec, launch, &output);
        if (valid != test.valid || (valid && output != test.expected) || (!valid && output != 0x42)) {
            std::fprintf(stderr, "FAIL %s valid=%d output=%llx\n", test.spec, valid, output);
            return 1;
        }
        ++count;
    }
    launch.params = nullptr;
    unsigned long long output = 0;
    if (prof::ResolveTapAddress("p0", launch, &output) || prof::ResolveTapAddress("p0", launch, nullptr)) return 1;
    std::printf("PASS %u parser cases and null guards\n", count);
    return 0;
}
