#include <pybind11/pybind11.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace py = pybind11;

class AsyncTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    std::thread query_thread;
    std::atomic<bool> running = false;
    std::atomic<bool> completed = false;
    double elapsed_ms = 0.0;
    mutable std::mutex mtx;
    std::condition_variable cv;

    void query_loop() {
        std::unique_lock<std::mutex> lock(mtx);
        while (running) {
            if (cv.wait_for(lock, std::chrono::microseconds(100)) == std::cv_status::timeout) {
                if (!completed) {
                    lock.unlock();
                    cudaError_t status = cudaEventQuery(end_event);
                    lock.lock();

                    if (status == cudaSuccess) {
                        float temp_ms = 0.0f;
                        cudaEventElapsedTime(&temp_ms, start_event, end_event);
                        elapsed_ms = static_cast<double>(temp_ms);
                        completed = true;
                    } else if (status != cudaErrorNotReady) {
                        elapsed_ms = -1.0;
                        completed = true;
                    }
                }
            }
        }
    }

public:
    AsyncTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
    }

    ~AsyncTimer() {
        stop();
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }

    // 无参版本（默认流）
    void start() {
        start(cudaStreamPerThread);  // 使用线程默认流
    }

    // 带流参数版本（接受cudaStream_t）
    void start(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx);
        completed = false;
        elapsed_ms = 0.0;
        cudaEventRecord(start_event, stream);

        if (!running) {
            running = true;
            query_thread = std::thread(&AsyncTimer::query_loop, this);
            query_thread.detach();
        }
    }

    // 无参版本（默认流）
    void end() {
        end(cudaStreamPerThread);  // 使用线程默认流
    }

    // 带流参数版本（接受cudaStream_t）
    void end(cudaStream_t stream) {
        cudaEventRecord(end_event, stream);
    }

    bool is_completed() const {
        return completed;
    }

    double get_elapsed() const {
        std::lock_guard<std::mutex> lock(mtx);
        return elapsed_ms;
    }

    void stop() {
        running = false;
        cv.notify_one();
        if (query_thread.joinable()) {
            query_thread.join();
        }
    }
};

PYBIND11_MODULE(async_timer, m) {
    m.doc() = "异步CUDA计时器（纯PyBind11实现，不依赖PyTorch）";
    py::class_<AsyncTimer>(m, "AsyncTimer")
        .def(py::init<>(), "创建计时器实例")
        // 绑定无参版本start（默认流）
        .def("start", py::overload_cast<>(&AsyncTimer::start), "开始计时（使用默认CUDA流）")
        // 绑定带流参数的start（通过整数指针传递）
        .def("start", 
             [](AsyncTimer &timer, uint64_t stream_ptr) {
                 timer.start(reinterpret_cast<cudaStream_t>(stream_ptr));
             }, 
             py::arg("stream_ptr"), "开始计时（通过指针指定CUDA流）")
        // 绑定无参版本end（默认流）
        .def("end", py::overload_cast<>(&AsyncTimer::end), "结束计时（使用默认CUDA流）")
        // 绑定带流参数的end（通过整数指针传递）
        .def("end", 
             [](AsyncTimer &timer, uint64_t stream_ptr) {
                 timer.end(reinterpret_cast<cudaStream_t>(stream_ptr));
             }, 
             py::arg("stream_ptr"), "结束计时（通过指针指定CUDA流）")
        .def("is_completed", &AsyncTimer::is_completed, "检查计时是否完成")
        .def("get_elapsed", &AsyncTimer::get_elapsed, "获取耗时（毫秒）")
        .def("stop", &AsyncTimer::stop, "停止计时器线程");
}
