// Crash-handler tests.
//
// Each case spawns the `crash_probe` helper (see crash_probe.cpp) with a crash
// type and a private report directory, then inspects the artifacts it left
// behind. The probe deliberately crashes a child process rather than this test
// process, so the whole matrix can run in one gtest binary.

#include <gtest/gtest.h>

#include "crash_scenario.hpp"

#include <csignal>

import std;
import std.compat;

#ifndef CRASH_PROBE_PATH
#error "CRASH_PROBE_PATH must be defined by the build system"
#endif

namespace {

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

struct ProbeRun {
    int status = 0;
    bool report_exists = false;
    std::string report;
    std::string session;
};

ProbeRun RunProbe(const std::string& type) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("vkengine_crash_test_" + type);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    const std::string command =
        std::string{"\""} + CRASH_PROBE_PATH + "\" " + type + " \"" + dir.string() + "\"";
    const int status = std::system(command.c_str());

    ProbeRun run;
    run.status = status;

    const std::filesystem::path report = dir / "crash_report.txt";
    run.report_exists = std::filesystem::exists(report, ec);
    if (run.report_exists) {
        run.report = ReadFile(report);
    }
    run.session = ReadFile(dir / "session.log");
    return run;
}

std::size_t CountOccurrences(const std::string& haystack, const std::string& needle) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

// Every fault report must preserve the probe's context: the app and stage
// values it was told to record, the breadcrumbs it emitted (as a count, not a
// pinned format), and the log tail (which still holds the marker line). The
// surrounding report wording is the engine's choice and is not asserted.
void ExpectCapturedContext(const ProbeRun& run) {
    ASSERT_TRUE(run.report_exists);
    EXPECT_NE(run.report.find("crash_probe"), std::string::npos);
    EXPECT_NE(run.report.find("probe stage"), std::string::npos);
    EXPECT_EQ(CountOccurrences(run.report, "probe-crumb-"),
              static_cast<std::size_t>(vkengine_test::crash_scenario::kCrumbs));
    EXPECT_NE(run.report.find("MARKER-LINE-12345"), std::string::npos);
    // The session log is the on-disk copy, written line by line as logging happens.
    EXPECT_NE(run.session.find("MARKER-LINE-12345"), std::string::npos);
}

// The report must carry the OS-reported signal number itself (on Windows the
// mapped structured-exception code); how it labels it is presentation.
void ExpectSignalReported(const ProbeRun& run, int expected_number) {
#ifdef _WIN32
    static_cast<void>(expected_number);
#else
    const std::string number = std::to_string(expected_number);
    EXPECT_NE(run.report.find(number), std::string::npos)
        << "report does not mention OS-reported signal number " << number;
#endif
}

} // namespace

TEST(CrashHandler, DoesNotReportOnCleanExit) {
    const ProbeRun run = RunProbe("none");
    EXPECT_EQ(run.status, 0);
    EXPECT_FALSE(run.report_exists);
    EXPECT_NE(run.session.find("MARKER-LINE-12345"), std::string::npos);
}

TEST(CrashHandler, ReportWritesReportWithoutTerminating) {
    const ProbeRun run = RunProbe("report");
    EXPECT_EQ(run.status, 0);
    ExpectCapturedContext(run);
    // The probe-supplied reason and detail values, not the surrounding wording.
    EXPECT_NE(run.report.find("probe report"), std::string::npos);
    EXPECT_NE(run.report.find("probe detail"), std::string::npos);
}

TEST(CrashHandler, FatalTerminatesAndReports) {
    const ProbeRun run = RunProbe("fatal");
    EXPECT_NE(run.status, 0);
    ExpectCapturedContext(run);
    EXPECT_NE(run.report.find("probe fatal"), std::string::npos);
}

class CrashTypeTest : public ::testing::TestWithParam<const char*> {};

TEST_P(CrashTypeTest, CapturesReportAndLogTail) {
    const ProbeRun run = RunProbe(GetParam());
    SCOPED_TRACE(GetParam());
    EXPECT_NE(run.status, 0);
    ExpectCapturedContext(run);
}

INSTANTIATE_TEST_SUITE_P(
    CrashTypes,
    CrashTypeTest,
    ::testing::Values("segv", "fpe", "ill", "abort", "terminate", "throw"
#ifdef SIGBUS
                      ,
                      "bus"
#endif
                      ));

TEST(CrashHandler, SegfaultReasonMatchesPlatform) {
    const ProbeRun run = RunProbe("segv");
    ASSERT_TRUE(run.report_exists);
    ExpectCapturedContext(run);
    ExpectSignalReported(run, static_cast<int>(SIGSEGV));
#ifdef _WIN32
    // The OS-reported structured-exception code for an access violation.
    EXPECT_NE(run.report.find("0xc0000005"), std::string::npos);
#endif
}

TEST(CrashHandler, AbortReasonMatchesPlatform) {
    const ProbeRun run = RunProbe("abort");
    ASSERT_TRUE(run.report_exists);
    ExpectCapturedContext(run);
    ExpectSignalReported(run, static_cast<int>(SIGABRT));
}

TEST(CrashHandler, UnhandledExceptionReportsTerminate) {
    const ProbeRun run = RunProbe("throw");
    ASSERT_TRUE(run.report_exists);
    // The report identifies the terminate path without pinning its wording.
    EXPECT_NE(run.report.find("terminate"), std::string::npos);
}

TEST(CrashHandler, ExplicitTerminateIsReported) {
    const ProbeRun run = RunProbe("terminate");
    ASSERT_TRUE(run.report_exists);
    EXPECT_NE(run.report.find("terminate"), std::string::npos);
}

// A large burst of lines must survive clean exit even though the session log is
// buffered and only flushed in batches.
TEST(CrashHandler, BulkLogIsCompleteOnCleanExit) {
    const ProbeRun run = RunProbe("bulk");
    EXPECT_EQ(run.status, 0);
    EXPECT_FALSE(run.report_exists);
    EXPECT_EQ(CountOccurrences(run.session, "BULK-"),
              static_cast<std::size_t>(vkengine_test::crash_scenario::kBulkLines));
    EXPECT_NE(run.session.find("BULK-000000"), std::string::npos);
    EXPECT_NE(run.session.find("BULK-004999"), std::string::npos);
}

// Lines still sitting in the buffer when the process faults must be drained by
// the handler, so nothing committed is lost.
TEST(CrashHandler, BulkLogIsCompleteAfterCrash) {
    const ProbeRun run = RunProbe("bulk-crash");
    EXPECT_NE(run.status, 0);
    EXPECT_EQ(CountOccurrences(run.session, "BULK-"),
              static_cast<std::size_t>(vkengine_test::crash_scenario::kBulkLines));
}

// Several threads are alive with unflushed buffers at fault time; the handler
// has to drain every one of them.
TEST(CrashHandler, FaultDrainsAllThreadBuffers) {
    const ProbeRun run = RunProbe("mt-crash");
    EXPECT_NE(run.status, 0);
    for (int id = 0; id < vkengine_test::crash_scenario::kMtThreads; ++id) {
        for (int i = 0; i < vkengine_test::crash_scenario::kMtLinesPerThread; ++i) {
            char line[64];
            std::snprintf(line, sizeof(line), "MT-%d-LINE-%02d", id, i);
            EXPECT_NE(run.session.find(line), std::string::npos) << line;
        }
    }
    EXPECT_EQ(CountOccurrences(run.session, "MT-"),
              static_cast<std::size_t>(vkengine_test::crash_scenario::kMtThreads *
                                       vkengine_test::crash_scenario::kMtLinesPerThread));
}

// A thread that logs and exits flushes its own buffer, independent of Shutdown.
TEST(CrashHandler, WorkerThreadFlushesOnExit) {
    const ProbeRun run = RunProbe("thread");
    EXPECT_EQ(run.status, 0);
    EXPECT_NE(run.session.find("THREAD-MARKER-1"), std::string::npos);
    EXPECT_NE(run.session.find("THREAD-MARKER-2"), std::string::npos);
    EXPECT_NE(run.session.find("THREAD-MARKER-3"), std::string::npos);
}

// A single line larger than a whole buffer must be written intact.
TEST(CrashHandler, LongLineIsWrittenIntact) {
    const ProbeRun run = RunProbe("long");
    EXPECT_EQ(run.status, 0);
    EXPECT_NE(run.session.find("LONG-BEGIN"), std::string::npos);
    EXPECT_NE(run.session.find("LONG-END"), std::string::npos);
    EXPECT_NE(run.session.find(
                  std::string(static_cast<std::size_t>(vkengine_test::crash_scenario::kLongLineBytes), 'x')),
              std::string::npos);
}

// An explicit Flush() makes the log current without waiting for a full buffer
// or for Shutdown.
TEST(CrashHandler, ExplicitFlushPersistsWithoutShutdown) {
    const ProbeRun run = RunProbe("flush");
    EXPECT_EQ(run.status, 0);
    EXPECT_NE(run.session.find("FLUSH-BEFORE-777"), std::string::npos);
    // The probe _Exit()s right after Flush(), which skips the destructor
    // flush, so a line logged after the flush must be absent.
    EXPECT_EQ(run.session.find("FLUSH-AFTER-888"), std::string::npos);
}
