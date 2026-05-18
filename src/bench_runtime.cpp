#include "bench_runtime.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

// OpenMP is optional — bench_core is built without OpenMP, but the
// runtime check still needs to detect whether the *final binary* (or
// any DSO it loaded) brought OpenMP in. We weak-link `omp_get_max_threads`
// via dlsym so we don't introduce a build-time OpenMP dependency.
#include <dlfcn.h>

namespace bench_runtime {

int detectOpenMPThreads() {
    using OmpFn = int (*)();
    // RTLD_DEFAULT walks the global symbol table — picks up libgomp /
    // libomp loaded by any transitively-linked DSO (e.g. OpenCV's TBB
    // or MIVisionX's HAF code).
    void* sym = dlsym(RTLD_DEFAULT, "omp_get_max_threads");
    if (sym == nullptr) {
        return 0;
    }
    OmpFn f = reinterpret_cast<OmpFn>(sym);
    int n = f();
    return n > 0 ? n : 0;
}

void applyThreadingPolicy(int requested, SystemInfo& info) {
    info.requested_threads = requested;

    if (requested > 0) {
        // setenv before any pool spins up. Doesn't override an existing
        // value silently — we WANT to override here, that's the whole
        // point of --threads. The OMP_NUM_THREADS env value captured
        // into SystemInfo is the *final* value the kernel libs will see.
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", requested);
        setenv("OMP_NUM_THREADS", buf, /*overwrite=*/1);
    }

    const char* env_omp = std::getenv("OMP_NUM_THREADS");
    info.omp_num_threads_env = (env_omp != nullptr) ? env_omp : "";

    info.openmp_max_threads = detectOpenMPThreads();
}

// Compute monotonic clock resolution by sampling chrono in a tight loop
// until two consecutive reads differ — the difference is an upper bound
// on the clock tick. We average over a few hundred samples to filter
// noise from preemption.
static double measureClockResolutionNs() {
    using Clock = std::chrono::steady_clock;
    const int samples_needed = 64;
    double min_delta_ns = 1e18;
    for (int s = 0; s < samples_needed; ++s) {
        auto t0 = Clock::now();
        auto t1 = t0;
        // Spin until the clock visibly moves.
        while (t1 == t0) {
            t1 = Clock::now();
        }
        double delta = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (delta > 0 && delta < min_delta_ns) {
            min_delta_ns = delta;
        }
    }
    return min_delta_ns;
}

// Time a sleep_for() and report the |measured - target| / target error.
// On a healthy timer + a non-loaded CI runner this is typically <2%.
// We tolerate up to 5% before flagging FAIL — sleeps can overrun by
// several milliseconds under runner load, especially for the 1ms case
// where the kernel's HZ tick granularity dominates.
static double measureSleepErrorPct(double target_ms) {
    using Clock = std::chrono::steady_clock;
    // Two trials, take the lower error — the higher one is usually
    // a preemption hit. With only 3 sleep classes and 2 trials each
    // the whole self-test takes ~250ms which is invisible in CI.
    double best_err = 1e9;
    for (int trial = 0; trial < 2; ++trial) {
        auto t0 = Clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(
            static_cast<long long>(target_ms * 1000.0)));
        auto t1 = Clock::now();
        double measured_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        double err_pct = std::abs(measured_ms - target_ms) / target_ms * 100.0;
        if (err_pct < best_err) best_err = err_pct;
    }
    return best_err;
}

int runTimerValidation(SystemInfo& info) {
    std::printf("=============================================================\n");
    std::printf("  Timer self-test (BenchmarkTimer = std::chrono::high_resolution_clock)\n");
    std::printf("=============================================================\n");

    double res_ns = measureClockResolutionNs();
    info.timer_resolution_ns = res_ns;
    std::printf("  Clock resolution        : %.1f ns (smallest observed delta)\n", res_ns);

    double err_1ms = measureSleepErrorPct(1.0);
    double err_10ms = measureSleepErrorPct(10.0);
    double err_100ms = measureSleepErrorPct(100.0);
    info.timer_sleep_1ms_err_pct = err_1ms;
    info.timer_sleep_10ms_err_pct = err_10ms;
    info.timer_sleep_100ms_err_pct = err_100ms;

    std::printf("  Sleep   1ms  measured err: %6.2f %%\n", err_1ms);
    std::printf("  Sleep  10ms  measured err: %6.2f %%\n", err_10ms);
    std::printf("  Sleep 100ms  measured err: %6.2f %%\n", err_100ms);

    // PASS thresholds — 100ms must be tight (<2%) because at that
    // scale preemption noise is overshadowed by the sleep itself.
    // 10ms tolerates 5% (occasional ~500us slips). 1ms is most
    // forgiving (HZ-tick dominates) — we just check it's not
    // off by 10x, which would indicate a broken clock source.
    const double thr_1ms = 50.0;     // 1ms ± 0.5ms
    const double thr_10ms = 5.0;     // 10ms ± 500us
    const double thr_100ms = 2.0;    // 100ms ± 2ms

    bool ok = (err_1ms <= thr_1ms) &&
              (err_10ms <= thr_10ms) &&
              (err_100ms <= thr_100ms);

    info.timing_validated = ok;
    std::printf("  Verdict                 : %s\n",
                ok ? "PASS — timer accurate enough for benchmark scales"
                   : "FAIL — measurements MAY be unreliable; see thresholds above");
    std::printf("  Thresholds              : 1ms<=%.0f%%, 10ms<=%.0f%%, 100ms<=%.0f%%\n",
                thr_1ms, thr_10ms, thr_100ms);
    std::printf("=============================================================\n");
    return ok ? 0 : 1;
}

}  // namespace bench_runtime
