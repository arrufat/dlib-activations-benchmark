#include <cstdlib>
#include <dlib/dnn.h>

namespace chrono = std::chrono;
using fus = chrono::duration<float, std::micro>;

template <typename fwd_func, typename bwd_func> auto benchmark(
    fwd_func fwd,
    bwd_func bwd,
    const std::string& name,
    int iterations = 100,
    int warmup = 10)
{
    dlib::resizable_tensor input, output, gradient;
    chrono::time_point<chrono::steady_clock> t0, t1;
    input.set_size(16, 10, 256, 256);
    output.copy_size(input);
    gradient.copy_size(input);
    t0 = chrono::steady_clock::now();
    dlib::running_stats<float> fwd_stats, bwd_stats;
    for (int i = 0; i < warmup; ++i)
    {
        fwd(output, input);
    }
    t1 = chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        t0 = chrono::steady_clock::now();
        fwd(output, input);
        t1 = chrono::steady_clock::now();
        fwd_stats.add(chrono::duration_cast<fus>(t1 - t0).count());
    }
    std::cout << name << " fwd: " << fwd_stats.mean() << " ± " << fwd_stats.stddev() << " µs\n";

    for (int i = 0; i < warmup; ++i)
    {
        bwd(gradient, output, input);
    }

    for (int i = 0; i < iterations; ++i)
    {
        t0 = chrono::steady_clock::now();
        bwd(gradient, output, input);
        t1 = chrono::steady_clock::now();
        bwd_stats.add(chrono::duration_cast<fus>(t1 - t0).count());
    }
    std::cout << name << " bwd: " << bwd_stats.mean() << " ± " << bwd_stats.stddev() << " µs\n";
}

auto main(/*const int argc, const char** argv*/) -> int
try
{
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    const int warmup = 10;
    const int iterations = 100;

    benchmark(dlib::tt::relu, dlib::tt::relu_gradient, "relu", warmup, iterations);
    benchmark(dlib::tt::mish, dlib::tt::mish_gradient, "mish", warmup, iterations);
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
