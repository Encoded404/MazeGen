#include "engine/core/bootstrap/Crash.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <mutex>
#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <signal.h>
#  include <unistd.h>
#  if __has_include(<execinfo.h>)
#    include <execinfo.h>
#    define VKENGINE_HAVE_EXECINFO 1
#  endif
#endif

// AddressSanitizer installs its own SIGSEGV/SIGBUS handlers and produces far
// better diagnostics than this subsystem can. When it is active (Debug builds
// with ENABLE_SANITIZERS), leave the signal handlers to ASan and only keep the
// crash ring + session log capture. Release builds have no ASan and get the
// full handler set.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define VKENGINE_ASAN_ACTIVE 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(VKENGINE_ASAN_ACTIVE)
#  define VKENGINE_ASAN_ACTIVE 1
#endif

namespace VulkanEngine::Crash {
namespace {

// ── Crash-safe log ring ────────────────────────────────────────────────
// One fixed slot per line: LogLine(), Stage() and Breadcrumb() are the only
// writers, and they never allocate. The fault handlers read the ring with
// plain open/write and never take a lock, so a crash while another thread
// holds the logging mutex is still capturable.
constexpr unsigned kRingLines = 1024;
constexpr unsigned kLineCap = 384;
constexpr unsigned kCrumbs = 64;

char g_ring[kRingLines][kLineCap];
std::atomic<unsigned> g_ring_len[kRingLines]{};
std::atomic<unsigned> g_ring_next{0};

char g_crumb_name[kCrumbs][64];
std::atomic<unsigned> g_crumb_name_len[kCrumbs]{};
std::atomic<unsigned long long> g_crumb_value[kCrumbs]{};
std::atomic<unsigned> g_crumb_next{0};

const char* const kStageStartup = "startup";
std::atomic<const char*> g_stage{kStageStartup};

std::atomic<bool> g_installed{false};
[[maybe_unused]] std::atomic<bool> g_faulted{false};
std::atomic<bool> g_reporting{false};

// ── Buffered session-log storage ───────────────────────────────────────
// Each logging thread owns one fixed buffer ("slot") and appends whole lines
// to it, draining it with a single write() once it crosses kLogThreshold. A
// per-slot 'busy' flag serialises appends and flushes for that one buffer;
// because a slot is owned by a single thread the fast path is effectively
// uncontended. The fault handlers drain every slot with write() only, so a
// crash still captures every committed line. Only whole lines are ever
// published, so a reader can never observe a torn line.
constexpr unsigned kMaxLogThreads = 64;
constexpr std::size_t kLogChunk = 32 * 1024;
constexpr std::size_t kLogThreshold = 24 * 1024;

struct LogSlot {
    std::atomic<char*> data{nullptr};
    std::atomic<unsigned> len{0};
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

LogSlot g_slots[kMaxLogThreads];
alignas(64) char g_log_pool[kMaxLogThreads][kLogChunk];
alignas(64) char g_overflow_buf[kLogChunk];
LogSlot g_overflow_slot;
std::atomic<unsigned> g_slot_high_water{0};

// Taken only while a thread claims or retires a slot, never on the per-line
// path.
std::mutex g_slot_mutex;
unsigned g_free_slots[kMaxLogThreads];
unsigned g_free_count = 0;

// Serialises the rare path that writes a single line longer than kLogChunk.
std::mutex g_long_line_mutex;

// Open session-log handle: a POSIX fd or a Win32 HANDLE stored as intptr_t.
// -1 means "not open". It is opened with append semantics so concurrent
// write() calls from several threads land atomically without a lock.
std::atomic<std::intptr_t> g_log_handle{-1};

char g_report_dir[1024] = ".";
char g_session_path[1200] = "";
char g_crash_path[1200] = "";
char g_app_name[128] = "MazeGenerator";
char g_boot_time[64] = "";

// ── Async-safe raw output ──────────────────────────────────────────────
#if defined(_WIN32)
struct RawOut {
    HANDLE handle = INVALID_HANDLE_VALUE;
    [[nodiscard]] bool valid() const { return handle != INVALID_HANDLE_VALUE; }
    void write(const void* data, std::size_t size) const {
        if (!valid() || size == 0) return;
        DWORD written = 0;
        ::WriteFile(handle, data, static_cast<DWORD>(size), &written, nullptr);
    }
    void close() {
        if (valid()) {
            ::CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
    }
};

RawOut RawOpen(const char* path) {
    RawOut out;
    if (path != nullptr && *path != '\0') {
        out.handle = ::CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                   FILE_ATTRIBUTE_NORMAL, nullptr);
    }
    return out;
}
#else
struct RawOut {
    int fd = -1;
    [[nodiscard]] bool valid() const { return fd >= 0; }
    void write(const void* data, std::size_t size) const {
        if (fd < 0 || size == 0) return;
        const ssize_t written = ::write(fd, data, size);
        static_cast<void>(written);
    }
    void close() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
};

RawOut RawOpen(const char* path) {
    RawOut out;
    if (path != nullptr && *path != '\0') {
        out.fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    }
    return out;
}
#endif

// ── Session-log handle ─────────────────────────────────────────────────
// A single pre-opened append handle. write()/WriteFile are async-signal-safe,
// so the fault handlers can flush buffered lines without a lock. The handle is
// never closed from a context that may race a writer: Shutdown() just marks it
// invalid and lets the OS reclaim it at process exit.

bool WriteHandle(std::intptr_t raw, const void* data, std::size_t size) noexcept {
    if (raw < 0 || data == nullptr || size == 0) {
        return false;
    }
    const char* bytes = static_cast<const char*>(data);
#if defined(_WIN32)
    const HANDLE handle = reinterpret_cast<HANDLE>(raw);
    while (size > 0) {
        const DWORD chunk = static_cast<DWORD>(size > 0x100000u ? 0x100000u : size);
        DWORD written = 0;
        if (::WriteFile(handle, bytes, chunk, &written, nullptr) == 0 || written == 0) {
            return false;
        }
        bytes += written;
        size -= written;
    }
#else
    const int fd = static_cast<int>(raw);
    while (size > 0) {
        const ssize_t written = ::write(fd, bytes, size);
        if (written > 0) {
            bytes += written;
            size -= static_cast<std::size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
#endif
    return true;
}

void WriteAll(std::intptr_t handle, const void* data, std::size_t size) noexcept {
    static_cast<void>(WriteHandle(handle, data, size));
}

std::intptr_t OpenSessionHandle(const char* path) noexcept {
    if (path == nullptr || *path == '\0') {
        return -1;
    }
#if defined(_WIN32)
    const HANDLE handle = ::CreateFileA(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                        nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    return reinterpret_cast<std::intptr_t>(handle);
#else
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
    return static_cast<std::intptr_t>(fd);
#endif
}

void CloseSessionHandle(std::intptr_t handle) noexcept {
#if defined(_WIN32)
    if (handle != -1) {
        ::CloseHandle(reinterpret_cast<HANDLE>(handle));
    }
#else
    if (handle >= 0) {
        ::close(static_cast<int>(handle));
    }
#endif
}

// ── Buffered session-log slot operations ───────────────────────────────

void LockSlot(LogSlot& slot) noexcept {
    while (slot.busy.test_and_set(std::memory_order_acquire)) {
        // Spin. A slot is owned by one thread, so this only ever contends with
        // a shutdown or fault drainer.
    }
}

bool TryLockSlot(LogSlot& slot) noexcept {
    // Bounded: the fault path must never block indefinitely on a slot owned by
    // a thread that is itself mid-operation (possibly the faulting thread).
    for (int attempt = 0; attempt < 1024; ++attempt) {
        if (!slot.busy.test_and_set(std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

void UnlockSlot(LogSlot& slot) noexcept {
    slot.busy.clear(std::memory_order_release);
}

// Flush whatever the slot already holds. Caller does not hold 'busy'.
void FlushSlot(LogSlot& slot) noexcept {
    LockSlot(slot);
    char* const buf = slot.data.load(std::memory_order_relaxed);
    const unsigned len = slot.len.load(std::memory_order_relaxed);
    if (buf != nullptr && len != 0) {
        WriteAll(g_log_handle.load(std::memory_order_relaxed), buf, len);
        slot.len.store(0, std::memory_order_relaxed);
    }
    UnlockSlot(slot);
}

struct TlsLog;
void ReleaseTlsLog(TlsLog& t) noexcept;

// Per-thread cursor into the slot registry. Registration happens once, on the
// first line a thread logs.
struct TlsLog {
    char* buf = nullptr;
    LogSlot* slot = nullptr;
    unsigned index = kMaxLogThreads;
    bool overflow = false;

    ~TlsLog() { ReleaseTlsLog(*this); }
};

thread_local TlsLog t_log;

void AcquireTlsLog(TlsLog& t) noexcept {
    try {
        std::lock_guard<std::mutex> const lock(g_slot_mutex);
        unsigned index = kMaxLogThreads;
        if (g_free_count > 0) {
            index = g_free_slots[--g_free_count];
        } else {
            const unsigned high_water = g_slot_high_water.load(std::memory_order_relaxed);
            if (high_water < kMaxLogThreads) {
                index = high_water;
                g_slot_high_water.store(high_water + 1, std::memory_order_release);
            }
        }
        if (index >= kMaxLogThreads) {
            // More live logging threads than slots: share the overflow buffer.
            // Lines still reach the file, just serialised through one slot.
            g_overflow_slot.data.store(g_overflow_buf, std::memory_order_release);
            t.overflow = true;
            t.slot = &g_overflow_slot;
            t.buf = g_overflow_buf;
            t.index = kMaxLogThreads;
            return;
        }
        t.overflow = false;
        t.index = index;
        t.slot = &g_slots[index];
        t.buf = g_log_pool[index];
        t.slot->len.store(0, std::memory_order_relaxed);
        t.slot->data.store(t.buf, std::memory_order_release);
    } catch (...) {
        // Registration must never break logging; fall back to the shared
        // overflow buffer.
        t.overflow = true;
        t.slot = &g_overflow_slot;
        t.buf = g_overflow_buf;
        t.index = kMaxLogThreads;
    }
}

TlsLog& Tls() noexcept {
    if (t_log.buf == nullptr) {
        AcquireTlsLog(t_log);
    }
    return t_log;
}

void AppendSlot(LogSlot& slot, const char* line, std::size_t length) noexcept {
    LockSlot(slot);
    char* const buf = slot.data.load(std::memory_order_relaxed);
    if (buf != nullptr) {
        const std::intptr_t handle = g_log_handle.load(std::memory_order_relaxed);
        unsigned len = slot.len.load(std::memory_order_relaxed);
        if (static_cast<std::size_t>(len) + length + 1 > kLogChunk) {
            if (len != 0) {
                WriteAll(handle, buf, len);
            }
            len = 0;
        }
        std::memcpy(buf + len, line, length);
        len += static_cast<unsigned>(length);
        buf[len++] = '\n';
        slot.len.store(len, std::memory_order_relaxed);
        if (len >= kLogThreshold) {
            WriteAll(handle, buf, len);
            slot.len.store(0, std::memory_order_relaxed);
        }
    }
    UnlockSlot(slot);
}

void AppendLongLine(const char* line, std::size_t length) noexcept {
    // Pathological only: a single line larger than a whole buffer. Flushed
    // whole under its own lock so it can never interleave with another chunk.
    try {
        std::lock_guard<std::mutex> const lock(g_long_line_mutex);
        const std::intptr_t handle = g_log_handle.load(std::memory_order_acquire);
        if (handle < 0) {
            return;
        }
        WriteAll(handle, line, length);
        WriteAll(handle, "\n", 1);
    } catch (...) {
    }
}

void AppendToSessionLog(const char* line, std::size_t length) noexcept {
    try {
        TlsLog& t = Tls();
        LogSlot& slot = t.overflow ? g_overflow_slot : *t.slot;
        if (length + 1 > kLogChunk) {
            FlushSlot(slot);
            AppendLongLine(line, length);
            return;
        }
        AppendSlot(slot, line, length);
    } catch (...) {
    }
}

void DrainSlotAsync(LogSlot& slot, std::intptr_t handle) noexcept {
    if (!TryLockSlot(slot)) {
        return;
    }
    char* const buf = slot.data.load(std::memory_order_relaxed);
    const unsigned len = slot.len.load(std::memory_order_relaxed);
    if (buf != nullptr && len != 0) {
        WriteAll(handle, buf, len);
    }
    UnlockSlot(slot);
}

void DrainAllAsync() noexcept {
    const std::intptr_t handle = g_log_handle.load(std::memory_order_acquire);
    if (handle < 0) {
        return;
    }
    const unsigned high_water = g_slot_high_water.load(std::memory_order_acquire);
    for (unsigned i = 0; i < high_water && i < kMaxLogThreads; ++i) {
        DrainSlotAsync(g_slots[i], handle);
    }
    DrainSlotAsync(g_overflow_slot, handle);
}

void FlushAllSlots() noexcept {
    if (g_log_handle.load(std::memory_order_acquire) < 0) {
        return;
    }
    const unsigned high_water = g_slot_high_water.load(std::memory_order_acquire);
    for (unsigned i = 0; i < high_water && i < kMaxLogThreads; ++i) {
        FlushSlot(g_slots[i]);
    }
    FlushSlot(g_overflow_slot);
}

void ReleaseTlsLog(TlsLog& t) noexcept {
    if (t.buf == nullptr) {
        return;
    }
    try {
        if (t.overflow) {
            FlushSlot(g_overflow_slot);
            t.buf = nullptr;
            return;
        }
        LogSlot& slot = *t.slot;
        LockSlot(slot);
        char* const buf = slot.data.load(std::memory_order_relaxed);
        const unsigned len = slot.len.load(std::memory_order_relaxed);
        if (buf != nullptr && len != 0) {
            WriteAll(g_log_handle.load(std::memory_order_relaxed), buf, len);
        }
        slot.len.store(0, std::memory_order_relaxed);
        slot.data.store(nullptr, std::memory_order_release);
        UnlockSlot(slot);
        {
            std::lock_guard<std::mutex> const lock(g_slot_mutex);
            if (g_free_count < kMaxLogThreads) {
                g_free_slots[g_free_count++] = t.index;
            }
        }
    } catch (...) {
    }
    t.buf = nullptr;
    t.slot = nullptr;
}

// ── Async-safe formatting (no printf) ──────────────────────────────────
void WriteStr(const RawOut& out, const char* text) {
    if (text == nullptr) return;
    out.write(text, std::strlen(text));
}

void WriteDec(const RawOut& out, unsigned long long value) {
    char reversed[24];
    int count = 0;
    if (value == 0) {
        reversed[count++] = '0';
    } else {
        while (value != 0 && count < 20) {
            reversed[count++] = static_cast<char>('0' + (value % 10));
            value /= 10;
        }
    }
    char text[24];
    for (int i = 0; i < count; ++i) {
        text[i] = reversed[count - 1 - i];
    }
    out.write(text, static_cast<std::size_t>(count));
}

void WriteHex(const RawOut& out, unsigned long long value) {
    static const char kDigits[] = "0123456789abcdef";
    char reversed[16];
    int count = 0;
    if (value == 0) {
        reversed[count++] = '0';
    } else {
        while (value != 0 && count < 16) {
            reversed[count++] = kDigits[value & 0xFu];
            value >>= 4;
        }
    }
    char text[16];
    for (int i = 0; i < count; ++i) {
        text[i] = reversed[count - 1 - i];
    }
    out.write(text, static_cast<std::size_t>(count));
}

void CopyString(char* dst, std::size_t cap, const char* src) {
    if (cap == 0) return;
    std::size_t i = 0;
    if (src != nullptr) {
        while (src[i] != '\0' && i + 1 < cap) {
            dst[i] = src[i];
            ++i;
        }
    }
    dst[i] = '\0';
}

// ── Stack capture ──────────────────────────────────────────────────────
void WriteStack(const RawOut& out) {
#if defined(_WIN32)
    void* frames[64];
    const USHORT count = ::RtlCaptureStackBackTrace(0, 64, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        WriteStr(out, "0x");
        WriteHex(out, reinterpret_cast<unsigned long long>(frames[i]));
        WriteStr(out, "\n");
    }
#elif defined(VKENGINE_HAVE_EXECINFO)
    void* frames[64];
    const int count = ::backtrace(frames, 64);
    if (count > 0) {
        ::backtrace_symbols_fd(frames, count, out.fd);
    }
#else
    static_cast<void>(out);
#endif
}

// ── Report body ────────────────────────────────────────────────────────
void WriteReportHeader(const RawOut& out, const char* reason) {
    WriteStr(out, "===== VULKANENGINE CRASH REPORT =====\n");
    WriteStr(out, "app: ");
    WriteStr(out, g_app_name);
    WriteStr(out, "\nreason: ");
    WriteStr(out, reason != nullptr ? reason : "unknown");
    WriteStr(out, "\nstarted: ");
    WriteStr(out, g_boot_time);
    WriteStr(out, "\nstage: ");
    WriteStr(out, g_stage.load(std::memory_order_relaxed));
    WriteStr(out, "\nsession log: ");
    WriteStr(out, g_session_path);
    WriteStr(out, "\n");
}

void WriteReportBody(const RawOut& out) {
    WriteStr(out, "\n--- breadcrumbs ---\n");
    const unsigned crumb_end = g_crumb_next.load(std::memory_order_relaxed);
    const unsigned crumb_begin = crumb_end > kCrumbs ? crumb_end - kCrumbs : 0;
    for (unsigned i = crumb_begin; i < crumb_end; ++i) {
        const unsigned slot = i % kCrumbs;
        const unsigned name_len = g_crumb_name_len[slot].load(std::memory_order_relaxed);
        if (name_len == 0) continue;
        WriteStr(out, g_crumb_name[slot]);
        WriteStr(out, " = ");
        WriteDec(out, g_crumb_value[slot].load(std::memory_order_relaxed));
        WriteStr(out, "\n");
    }

    WriteStr(out, "\n--- log tail ---\n");
    const unsigned ring_end = g_ring_next.load(std::memory_order_relaxed);
    const unsigned ring_begin = ring_end > kRingLines ? ring_end - kRingLines : 0;
    for (unsigned i = ring_begin; i < ring_end; ++i) {
        const unsigned slot = i % kRingLines;
        const unsigned len = g_ring_len[slot].load(std::memory_order_acquire);
        if (len == 0 || len >= kLineCap) continue;
        out.write(g_ring[slot], len);
        out.write("\n", 1);
    }

    WriteStr(out, "\n--- stack ---\n");
    WriteStack(out);
    WriteStr(out, "\n===== END CRASH REPORT =====\n");
}

// ── Path helpers ───────────────────────────────────────────────────────
std::string ExecutablePath() {
#if defined(_WIN32)
    char buffer[MAX_PATH * 4];
    const DWORD len = ::GetModuleFileNameA(nullptr, buffer, static_cast<DWORD>(sizeof(buffer)));
    if (len > 0 && len < sizeof(buffer)) return std::string(buffer, buffer + len);
#elif defined(__linux__)
    char buffer[4096];
    const ssize_t len = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) return std::string(buffer, buffer + len);
#endif
    return {};
}

void EnsureDirectory(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    static_cast<void>(ec);
}

void DeriveDefaultPaths() {
    const std::string exe = ExecutablePath();
    std::string exe_dir;
    if (!exe.empty()) {
        exe_dir = std::filesystem::path(exe).parent_path().string();
    }
    if (exe_dir.empty()) exe_dir = ".";

    const std::string dir = exe_dir + "/logs/" + g_app_name;
    EnsureDirectory(dir);
    CopyString(g_report_dir, sizeof(g_report_dir), dir.c_str());
    CopyString(g_session_path, sizeof(g_session_path), (dir + "/session.log").c_str());
    CopyString(g_crash_path, sizeof(g_crash_path), (dir + "/crash_report.txt").c_str());
}

void OpenSessionLog() noexcept {
    g_log_handle.store(OpenSessionHandle(g_session_path), std::memory_order_release);
}

// Close the current handle. Only ever called from single-threaded
// reconfiguration (SetReportDirectory), after all buffers have been drained,
// so no writer can still hold it.
void CloseSessionLog() noexcept {
    const std::intptr_t handle = g_log_handle.exchange(-1, std::memory_order_acq_rel);
    CloseSessionHandle(handle);
}

// ── Terminate / abort handling ─────────────────────────────────────────
void WriteFaultReport(const char* reason, unsigned long long code, const char* detail) {
    RawOut out = RawOpen(g_crash_path);
    if (!out.valid()) return;
    WriteReportHeader(out, reason);
    WriteStr(out, "code: 0x");
    WriteHex(out, code);
    WriteStr(out, "\n");
    if (detail != nullptr && *detail != '\0') {
        WriteStr(out, "detail: ");
        WriteStr(out, detail);
        WriteStr(out, "\n");
    }
    WriteReportBody(out);
    out.close();
}

void CrashTerminateHandler() {
    DrainAllAsync();
    WriteFaultReport("std::terminate", 0, "unhandled exception or noexcept violation");
    std::_Exit(3);
}

#if defined(_WIN32)
LONG WINAPI CrashExceptionFilter(EXCEPTION_POINTERS* info) {
    if (g_faulted.exchange(true)) {
        ::TerminateProcess(::GetCurrentProcess(), 3);
    }
    const unsigned long code =
        (info != nullptr && info->ExceptionRecord != nullptr) ? info->ExceptionRecord->ExceptionCode : 0u;
    DrainAllAsync();
    RawOut out = RawOpen(g_crash_path);
    if (out.valid()) {
        WriteReportHeader(out, "structured exception");
        WriteStr(out, "code: 0x");
        WriteHex(out, code);
        WriteStr(out, "\n");
        if (info != nullptr && info->ExceptionRecord != nullptr) {
            WriteStr(out, "fault address: 0x");
            WriteHex(out, reinterpret_cast<unsigned long long>(info->ExceptionRecord->ExceptionAddress));
            WriteStr(out, "\n");
        }
        WriteReportBody(out);
        out.close();
    }
    ::TerminateProcess(::GetCurrentProcess(), static_cast<UINT>(code != 0 ? code : 3u));
    return EXCEPTION_EXECUTE_HANDLER;
}

extern "C" void CrashAbortHandler(int) {
    DrainAllAsync();
    WriteFaultReport("SIGABRT", 0, nullptr);
    std::_Exit(3);
}
#else
#if !defined(VKENGINE_ASAN_ACTIVE)
extern "C" void CrashSignalHandler(int sig, siginfo_t* info, void* ucontext) {
    static_cast<void>(ucontext);
    if (g_faulted.exchange(true)) {
        ::_exit(128 + sig);
    }
    DrainAllAsync();
    RawOut out = RawOpen(g_crash_path);
    if (out.valid()) {
        WriteReportHeader(out, "signal");
        WriteStr(out, "signal: ");
        WriteDec(out, static_cast<unsigned long long>(sig));
        WriteStr(out, "\n");
        if (info != nullptr) {
            WriteStr(out, "fault address: 0x");
            WriteHex(out, reinterpret_cast<unsigned long long>(info->si_addr));
            WriteStr(out, "\n");
        }
        WriteReportBody(out);
        out.close();
    }
    // Re-raise with the default disposition so the OS still produces a core
    // dump and reports the conventional 128+N status.
    ::signal(sig, SIG_DFL);
    ::raise(sig);
    ::_exit(128 + sig);
}
#endif
#endif

void InstallHandlers() {
    std::set_terminate(&CrashTerminateHandler);
#if defined(_WIN32)
    ::SetUnhandledExceptionFilter(&CrashExceptionFilter);
    ::signal(SIGABRT, &CrashAbortHandler);
#  if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#  endif
#else
#if !defined(VKENGINE_ASAN_ACTIVE)
    static char altstack[64 * 1024];
    stack_t stack{};
    stack.ss_sp = altstack;
    stack.ss_size = sizeof(altstack);
    stack.ss_flags = 0;
    ::sigaltstack(&stack, nullptr);

    int signals[8];
    int count = 0;
    signals[count++] = SIGSEGV;
    signals[count++] = SIGBUS;
    signals[count++] = SIGILL;
    signals[count++] = SIGFPE;
    signals[count++] = SIGABRT;
#  ifdef SIGSYS
    signals[count++] = SIGSYS;
#  endif
#  ifdef SIGTRAP
    signals[count++] = SIGTRAP;
#  endif

    for (int i = 0; i < count; ++i) {
        struct sigaction action {};
        action.sa_sigaction = &CrashSignalHandler;
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        ::sigemptyset(&action.sa_mask);
        ::sigaction(signals[i], &action, nullptr);
    }
#endif
#endif
}

void RestoreHandlers() {
#if defined(_WIN32)
    ::SetUnhandledExceptionFilter(nullptr);
    ::signal(SIGABRT, SIG_DFL);
#else
#if !defined(VKENGINE_ASAN_ACTIVE)
    ::signal(SIGSEGV, SIG_DFL);
    ::signal(SIGBUS, SIG_DFL);
    ::signal(SIGILL, SIG_DFL);
    ::signal(SIGFPE, SIG_DFL);
    ::signal(SIGABRT, SIG_DFL);
#  ifdef SIGSYS
    ::signal(SIGSYS, SIG_DFL);
#  endif
#  ifdef SIGTRAP
    ::signal(SIGTRAP, SIG_DFL);
#  endif
#endif
#endif
}

} // namespace

// ── Public API ─────────────────────────────────────────────────────────

void Boot(const char* app_name) noexcept {
    if (g_installed.exchange(true)) {
        return;
    }
    try {
        if (app_name != nullptr && *app_name != '\0') {
            CopyString(g_app_name, sizeof(g_app_name), app_name);
        } else {
            const std::string exe = ExecutablePath();
            if (!exe.empty()) {
                const std::string stem = std::filesystem::path(exe).stem().string();
                if (!stem.empty()) CopyString(g_app_name, sizeof(g_app_name), stem.c_str());
            }
        }

        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32)
        ::localtime_s(&tm, &now);
#else
        ::localtime_r(&now, &tm);
#endif
        if (std::strftime(g_boot_time, sizeof(g_boot_time), "%Y-%m-%d %H:%M:%S", &tm) == 0) {
            CopyString(g_boot_time, sizeof(g_boot_time), "unknown");
        }

        DeriveDefaultPaths();
        OpenSessionLog();
        InstallHandlers();
    } catch (...) {
        // Crash handling must never abort startup. Handlers may be partially
        // installed; the app still runs.
    }
}

void Shutdown() noexcept {
    if (!g_installed.exchange(false)) {
        return;
    }
    FlushAllSlots();
    RestoreHandlers();
    // Mark the log closed without closing the handle: a concurrent writer may
    // have loaded it just before this store, and closing underneath it would
    // race. The OS reclaims the handle at process exit.
    g_log_handle.store(-1, std::memory_order_release);
}

void SetReportDirectory(const char* directory) noexcept {
    if (!g_installed.load()) {
        return;
    }
    try {
        // The directory Boot() chose before the command line was known, i.e.
        // beside the executable. It has served its purpose once the real
        // location is known.
        const std::string previous_dir = g_report_dir;

        FlushAllSlots();
        CloseSessionLog();
        if (directory != nullptr && *directory != '\0') {
            const std::string dir = directory;
            EnsureDirectory(dir);
            CopyString(g_report_dir, sizeof(g_report_dir), dir.c_str());
            CopyString(g_session_path, sizeof(g_session_path), (dir + "/session.log").c_str());
            CopyString(g_crash_path, sizeof(g_crash_path), (dir + "/crash_report.txt").c_str());
        } else {
            DeriveDefaultPaths();
        }
        OpenSessionLog();

        // remove() refuses a non-empty directory, so this can never destroy a
        // previous run's session log or crash report - it only takes away an
        // unused, empty directory, which is what the executable-relative
        // default leaves behind on every launch.
        if (previous_dir != g_report_dir) {
            std::error_code ec;
            std::filesystem::remove(previous_dir, ec);
        }
    } catch (...) {
        // Keep the previous directory on failure.
    }
}

void LogLine(const char* line) noexcept {
    if (line == nullptr) return;

    const std::size_t length = std::strlen(line);
    const unsigned index = g_ring_next.fetch_add(1, std::memory_order_relaxed);
    const unsigned slot = index % kRingLines;
    const std::size_t copy = length < (kLineCap - 1) ? length : (kLineCap - 1);
    if (copy > 0) {
        std::memcpy(g_ring[slot], line, copy);
    }
    g_ring[slot][copy] = '\0';
    g_ring_len[slot].store(static_cast<unsigned>(copy), std::memory_order_release);

    AppendToSessionLog(line, length);
}

void Flush() noexcept {
    try {
        if (t_log.buf == nullptr) {
            return;
        }
        FlushSlot(t_log.overflow ? g_overflow_slot : *t_log.slot);
    } catch (...) {
    }
}

void Stage(const char* name) noexcept {
    if (name == nullptr) return;
    g_stage.store(name, std::memory_order_relaxed);
}

void Breadcrumb(const char* name, unsigned long long value) noexcept {
    if (name == nullptr) return;
    const unsigned index = g_crumb_next.fetch_add(1, std::memory_order_relaxed);
    const unsigned slot = index % kCrumbs;
    g_crumb_value[slot].store(value, std::memory_order_relaxed);
    const std::size_t length = std::strlen(name);
    const std::size_t copy = length < (sizeof(g_crumb_name[slot]) - 1)
                                 ? length
                                 : (sizeof(g_crumb_name[slot]) - 1);
    if (copy > 0) {
        std::memcpy(g_crumb_name[slot], name, copy);
    }
    g_crumb_name[slot][copy] = '\0';
    g_crumb_name_len[slot].store(static_cast<unsigned>(copy), std::memory_order_release);
}

void Report(const char* reason, const char* detail) noexcept {
    if (g_reporting.exchange(true)) {
        return;
    }
    try {
        char line[512];
        const int written = std::snprintf(line, sizeof(line), "[CRASH] %s%s%s",
                                          reason != nullptr ? reason : "unknown",
                                          (detail != nullptr && *detail != '\0') ? ": " : "",
                                          (detail != nullptr && *detail != '\0') ? detail : "");
        if (written > 0) {
            LogLine(line);
        }
        FlushAllSlots();
        WriteFaultReport(reason != nullptr ? reason : "unknown", 0, detail);
    } catch (...) {
        // Nothing sensible left to do.
    }
}

void Fatal(const char* reason, const char* detail) noexcept {
    Report(reason, detail);
    Shutdown();
    std::_Exit(1);
}

bool Installed() noexcept {
    return g_installed.load(std::memory_order_relaxed);
}

const char* ReportDirectory() noexcept {
    return g_report_dir;
}

} // namespace VulkanEngine::Crash
