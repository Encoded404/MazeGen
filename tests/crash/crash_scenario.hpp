// Shared crash-probe scenario sizing.
//
// The probe (crash_probe.cpp) emits these log volumes and the tests
// (crash_handler_tests.cpp) assert that every line survived; both sides must
// agree on the numbers, so they live here once.
//
// These are scenario sizes chosen to stress the session-log paths (buffered
// bulk lines, several live thread buffers, one oversized line) — they are not
// engine constants.

#pragma once

namespace vkengine_test::crash_scenario {

// Lines in the "BULK-000000"... bursts of the bulk scenarios (clean-exit and
// crash-time variants).
inline constexpr int kBulkLines = 5000;

// Worker threads that stay alive with unflushed buffers until the fault.
inline constexpr int kMtThreads = 4;
// Buffered lines each of those threads commits before the fault
// ("MT-<id>-LINE-<i>"...).
inline constexpr int kMtLinesPerThread = 10;

// Bytes of the deliberately oversized single log line.
inline constexpr int kLongLineBytes = 40000;

// Distinct breadcrumbs the probe records before crashing; the tests assert the
// report preserves exactly this many.
inline constexpr int kCrumbs = 3;

} // namespace vkengine_test::crash_scenario