// Crash-handler probe.
//
// A tiny standalone executable used by crash_handler_tests. It links
// Crash.cpp directly (not the engine) so it can be built without sanitizers:
// the tests must exercise the real signal / structured-exception handlers,
// which AddressSanitizer would otherwise own in Debug builds.
//
// Usage: crash_probe <type> <report-dir>
//
// It always emits a known stage, breadcrumb and log marker before crashing, so
// the tests can assert that all of that context reached the crash report.

#include <csignal>

#include "engine/core/bootstrap/Crash.hpp"
#include "crash_scenario.hpp"

import std;
import std.compat;

namespace {

constexpr const char* kMarker = "[probe] MARKER-LINE-12345";

void EmitContext() {
    VulkanEngine::Crash::Stage("probe stage");
    VulkanEngine::Crash::LogLine(kMarker);
    for (int i = 1; i <= vkengine_test::crash_scenario::kCrumbs; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "probe-crumb-%d", i);
        VulkanEngine::Crash::Breadcrumb(name, static_cast<unsigned long long>(i));
    }
}

[[noreturn]] void TriggerCrash(const std::string& type) {
    if (type == "segv") {
        volatile int* null_pointer = nullptr;
        *null_pointer = 1;
    } else if (type == "fpe") {
        volatile int zero = 0;
        volatile int one = 1;
        volatile int quotient = one / zero;
        static_cast<void>(quotient);
    } else if (type == "ill") {
        __builtin_trap();
    }
#ifdef SIGBUS
    else if (type == "bus") {
        std::raise(SIGBUS);
    }
#endif
    else if (type == "abort") {
        std::abort();
    } else if (type == "terminate") {
        std::terminate();
    } else if (type == "throw") {
        throw std::runtime_error("probe thrown exception");
    }
    std::_Exit(90); // Unknown type: fail loudly rather than run on.
}

void LogBulk(int count, const char* prefix) {
    for (int i = 0; i < count; ++i) {
        char line[64];
        std::snprintf(line, sizeof(line), "%s-%06d", prefix, i);
        VulkanEngine::Crash::LogLine(line);
    }
}

void LogLongLine() {
    std::string text = "LONG-BEGIN";
    text.append(static_cast<std::size_t>(vkengine_test::crash_scenario::kLongLineBytes), 'x');
    text += "LONG-END";
    VulkanEngine::Crash::LogLine(text.c_str());
}

void RunWorkerThread() {
    std::thread worker([] {
        VulkanEngine::Crash::LogLine("THREAD-MARKER-1");
        VulkanEngine::Crash::LogLine("THREAD-MARKER-2");
    });
    worker.join();
    VulkanEngine::Crash::LogLine("THREAD-MARKER-3");
}

// Crash while several threads are alive and hold unflushed buffered lines, so
// the fault handler has to drain every thread's buffer.
[[noreturn]] void CrashWithLiveThreads() {
    const int kThreadCount = vkengine_test::crash_scenario::kMtThreads;
    std::atomic<int> ready{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);
    for (int id = 0; id < kThreadCount; ++id) {
        workers.emplace_back([id, &ready, &release] {
            for (int i = 0; i < vkengine_test::crash_scenario::kMtLinesPerThread; ++i) {
                char line[64];
                std::snprintf(line, sizeof(line), "MT-%d-LINE-%02d", id, i);
                VulkanEngine::Crash::LogLine(line);
            }
            ready.fetch_add(1);
            while (!release.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    while (ready.load() != kThreadCount) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    TriggerCrash("segv");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::puts("usage: crash_probe <type> <report-dir>");
        return 2;
    }
    const std::string type = argv[1];

    VulkanEngine::Crash::Boot("crash_probe");
    VulkanEngine::Crash::SetReportDirectory(argv[2]);
    EmitContext();

    if (type == "none") {
        return 0;
    }
    if (type == "report") {
        VulkanEngine::Crash::Report("probe report", "probe detail");
        return 0;
    }
    if (type == "fatal") {
        VulkanEngine::Crash::Fatal("probe fatal", "probe detail");
    }
    if (type == "bulk") {
        LogBulk(vkengine_test::crash_scenario::kBulkLines, "BULK");
        return 0;
    }
    if (type == "bulk-crash") {
        LogBulk(vkengine_test::crash_scenario::kBulkLines, "BULK");
        TriggerCrash("segv");
    }
    if (type == "long") {
        LogLongLine();
        return 0;
    }
    if (type == "flush") {
        // _Exit skips thread-local destructors, so only the explicit Flush()
        // can have persisted FLUSH-BEFORE; FLUSH-AFTER must be absent.
        VulkanEngine::Crash::LogLine("FLUSH-BEFORE-777");
        VulkanEngine::Crash::Flush();
        VulkanEngine::Crash::LogLine("FLUSH-AFTER-888");
        std::_Exit(0);
    }
    if (type == "thread") {
        RunWorkerThread();
        return 0;
    }
    if (type == "mt-crash") {
        CrashWithLiveThreads();
    }

    TriggerCrash(type);
}
