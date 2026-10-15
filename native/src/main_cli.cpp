// Command-line entry point: adb push to /data/local/tmp and run from `adb shell`.
#include <cstdio>

#include "membench.h"

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    mb::Options o;
    std::string err;
    if (!mb::parseArgs(args, o, err)) {
        fprintf(stderr, "%s\n\n%s", err.c_str(), mb::usage().c_str());
        return 2;
    }
    if (o.help) {
        fputs(mb::usage().c_str(), stdout);
        return 0;
    }
    return mb::run(o, [](const std::string& line) {
        fputs(line.c_str(), stdout);
        fputc('\n', stdout);
        fflush(stdout);
    });
}
