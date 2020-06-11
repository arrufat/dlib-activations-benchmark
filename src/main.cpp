#include <cstdlib>
#include <dlib/dnn.h>

namespace chrono = std::chrono;
using fus = chrono::duration<float, std::micro>;

auto main(/*const int argc, const char** argv*/) -> int
try
{
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    chrono::time_point<chrono::steady_clock> t0, t1;
    const int warmup = 10;
    const int iterations = 100;

    // setup the input and output tensors
    dlib::resizable_tensor input, output, gradient;
    input.set_size(64, 3, 224, 224);
    output.copy_size(input);
    gradient.copy_size(input);

    // fill the input tensor with random values
    auto rnd = dlib::tt::tensor_rand(0);
    rnd.fill_gaussian(input);

    dlib::running_stats<float> rs;
    for (int i = 0; i < warmup; ++i)
    {
        dlib::tt::relu(output, input);
    }
    for (int i = 0; i < iterations; ++i)
    {
        t0 = chrono::steady_clock::now();
        dlib::tt::relu(output, input);
        t1 = chrono::steady_clock::now();
        rs.add(chrono::duration_cast<fus>(t1 - t0).count());
    }
    std::cout << "relu fwd: " << rs.mean() << " ± " << rs.stddev() << " us\n";
    rs.clear();
    for (int i = 0; i < warmup; ++i)
    {
        dlib::tt::relu_gradient(gradient, output, input);
    }
    for (int i = 0; i < iterations; ++i)
    {
        t0 = chrono::steady_clock::now();
        dlib::tt::relu_gradient(gradient, output, input);
        t1 = chrono::steady_clock::now();
        rs.add(chrono::duration_cast<fus>(t1 - t0).count());
    }
    std::cout << "relu bwd: " << rs.mean() << " ± " << rs.stddev() << " us\n";

    rs.clear();
    for (int i = 0; i < warmup; ++i)
    {
        dlib::tt::mish(output, input);
    }
    for (int i = 0; i < iterations; ++i)
    {
        t0 = chrono::steady_clock::now();
        dlib::tt::mish(output, input);
        t1 = chrono::steady_clock::now();
        rs.add(chrono::duration_cast<fus>(t1 - t0).count());
    }
    std::cout << "mish fwd: " << rs.mean() << " ± " << rs.stddev() << " us\n";
    rs.clear();
    for (int i = 0; i < warmup; ++i)
    {
        dlib::tt::mish_gradient(gradient, output, input);
    }
    for (int i = 0; i < iterations; ++i)
    {
        t0 = chrono::steady_clock::now();
        dlib::tt::mish_gradient(gradient, output, input);
        t1 = chrono::steady_clock::now();
        rs.add(chrono::duration_cast<fus>(t1 - t0).count());
    }
    std::cout << "mish bwd: " << rs.mean() << " ± " << rs.stddev() << " us\n";

}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
