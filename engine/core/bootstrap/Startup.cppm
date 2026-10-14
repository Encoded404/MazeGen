module;

// logging_macros.hpp has no <memory> include, safe in GMF.
#include <logging/logging_macros.hpp>

// localtime_r/localtime_s are POSIX/Win32, not part of the std module.
#include <time.h>

// Crash.hpp includes no standard headers either: safe in the GMF.
#include "engine/core/bootstrap/Crash.hpp"

export module VulkanEngine.Startup;

import std;

import logiface;
import logiface.ConsoleLogger;

namespace {

// Tee sink: the console output plus the crash-captured ring and session log.
// The crash line is formatted the same way the console logger formats it, so a
// crash report reads exactly like the terminal output around it. Formatting
// only happens per emitted record (after the level gate), never per frame.
struct TeeLogger : public Logiface::Logger {
    Logiface::ConsoleLogger console{Logiface::Level::info, std::cout, std::cerr};
    std::mutex mutex{};

    void Log(const Logiface::Record& record) override {
        console.Log(record);

        std::lock_guard<std::mutex> const lock(mutex);
        const std::time_t time = std::chrono::system_clock::to_time_t(record.timestamp);
        std::tm local{};
#if defined(_WIN32)
        localtime_s(&local, &time);
#else
        localtime_r(&time, &local);
#endif
        std::array<char, 32> stamp{};
        if (std::strftime(stamp.data(), stamp.size(), "%F %T", &local) == 0) {
            stamp[0] = '\0';
        }

        std::string line;
        line.reserve(record.message.size() + 64);
        line += '[';
        line += stamp.data();
        line += "] ";
        line += Logiface::ToString(record.lvl);
        line += " (";
        line += record.location;
        line += ':';
        line += std::to_string(record.line);
        line += "): ";
        line += record.message;
        VulkanEngine::Crash::LogLine(line.c_str());
    }

    void SetLevel(Logiface::Level level) noexcept override { console.SetLevel(level); }
    [[nodiscard]] Logiface::Level GetLevel() const noexcept override { return console.GetLevel(); }
};

} // namespace

export namespace VulkanEngine::Startup {

[[nodiscard]] Logiface::Level ParseLogLevel(std::string_view level) {
    if (level == "trace") return Logiface::Level::trace;
    if (level == "debug") return Logiface::Level::debug;
    if (level == "info") return Logiface::Level::info;
    if (level == "warn") return Logiface::Level::warn;
    if (level == "error") return Logiface::Level::error;
    if (level == "critical") return Logiface::Level::critical;
    return Logiface::Level::info;
}

void InitializeLogger(Logiface::Level level) {
    // Logger is a singleton with static lifetime - raw pointer is fine. The
    // tee is created once and only its level is adjusted on later calls, so a
    // level change from the CLI never detaches the crash sink.
    static TeeLogger app_logger; // NOLINT(misc-const-correctness)
    app_logger.SetLevel(level);
    Logiface::SetLogger(&app_logger);
}

void InitializeLogger(std::string_view level) {
    InitializeLogger(ParseLogLevel(level));
}

} // namespace VulkanEngine::Startup
