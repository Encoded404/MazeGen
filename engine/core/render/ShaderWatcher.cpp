module;

#include <logging/logging_macros.hpp>

#include <efsw/efsw.hpp>

module VulkanEngine.ShaderWatcher;

import std;
import std.compat;

import logiface;

import VulkanEngine.ShaderManager;
import VulkanShared.Teardown;

namespace VulkanEngine::ShaderSystem {

namespace {

struct WatcherListener final : efsw::FileWatchListener {
    ShaderWatcher& owner;

    explicit WatcherListener(ShaderWatcher& owner_ref)
        : owner(owner_ref)
    {}

    void handleFileAction(efsw::WatchID /*watchid*/, const std::string& /*dir*/,
                          const std::string& filename, efsw::Action action,
                          const std::string& /*oldFilename*/) override {
        if (action != efsw::Actions::Modified && action != efsw::Actions::Add) return;
        if (!filename.ends_with(".slang")) return; // ignore temp-file artifacts
        owner.OnFileChanged(filename);
    }

    // efsw 1.7: native queues can drop events (inotify overflow, atomic
    // replace on some platforms). Recover by scheduling every shader source
    // in the watched directory; the debounce and the on-disk mtime check in
    // ReloadPath collapse unchanged files, so this is cheap in the common case.
    void handleMissedFileActions(efsw::WatchID /*watchid*/, const std::string& dir) override {
        std::error_code ec;
        const std::filesystem::directory_iterator it{dir, ec};
        if (ec) return;
        for (const auto& entry : it) {
            if (!entry.is_regular_file(ec)) continue;
            const std::string name = entry.path().filename().string();
            if (name.ends_with(".slang")) {
                owner.OnFileChanged(name);
            }
        }
    }
};

// inotify reports several events per save (write, chmod, close), and editors
// that save atomically may write the file twice (temp file + rename). Debounce
// so a single save collapses into exactly one reload after the burst settles.
inline constexpr auto kReloadDebounce = std::chrono::milliseconds(200);

} // anonymous namespace

struct ShaderWatcher::Impl {
    // Destroyed in reverse declaration order: the watcher (joining its thread)
    // is destroyed before the listener.
    std::unique_ptr<WatcherListener> listener;
    std::unique_ptr<efsw::FileWatcher> watcher;
};

ShaderWatcher::ShaderWatcher(ShaderManager& shaders)
    : shaders_(shaders)
{}

ShaderWatcher::~ShaderWatcher() {
    Stop();
}

void ShaderWatcher::Start() {
    if (impl_) return;
    impl_ = std::make_unique<Impl>();
    impl_->listener = std::make_unique<WatcherListener>(*this);
    impl_->watcher = std::make_unique<efsw::FileWatcher>(); // native backend (inotify on Linux)

    // Re-watch every directory remembered from before Start()/a previous run,
    // then pick up everything the shader manager currently knows about.
    std::vector<std::string> known;
    {
        std::scoped_lock lock(mutex_);
        known.assign(registered_dirs_.begin(), registered_dirs_.end());
        registered_dirs_.clear();
    }
    for (const auto& dir : known) {
        (void)AddDirectory(dir);
    }
    Refresh();

    impl_->watcher->watch();

    stop_.store(false, std::memory_order_release);
    debounce_thread_ = std::thread([this] { DebounceLoop(); });
}

bool ShaderWatcher::AddDirectory(const std::string& directory) {
    if (directory.empty()) {
        return false;
    }

    std::scoped_lock lock(mutex_);
    if (registered_dirs_.contains(directory)) {
        return true;
    }
    if (!impl_ || !impl_->watcher) {
        // Not started yet: remember it so Start() watches it.
        registered_dirs_.insert(directory);
        return true;
    }

    const auto watch_id = impl_->watcher->addWatch(directory, impl_->listener.get(), /*recursive=*/false);
    if (watch_id < 0) {
        LOGIFACE_LOG(warn, "ShaderWatcher: failed to watch " + directory);
        return false;
    }
    registered_dirs_.insert(directory);
    LOGIFACE_LOG(info, "ShaderWatcher: watching " + directory);
    return true;
}

void ShaderWatcher::Refresh() {
    for (const auto& dir : shaders_.GetSlangDirectories()) {
        (void)AddDirectory(dir);
    }
}

std::size_t ShaderWatcher::GetRegisteredDirectoryCount() const {
    std::scoped_lock lock(mutex_);
    return registered_dirs_.size();
}

void ShaderWatcher::Stop() {
    Quiesce();
    impl_.reset(); // ~FileWatcher stops and joins its internal thread
}

void ShaderWatcher::StopAsync() {
    Quiesce();
    // Quiesce() guarantees no thread can reach `shaders_` anymore, so dropping
    // the efsw watcher (whose internal thread join can block for its poll
    // interval) on a detached thread is safe: the join owns nothing else.
    auto orphan = std::move(impl_);
    if (orphan) {
        VulkanShared::RunDetached([orphan = std::move(orphan)]() mutable { orphan.reset(); });
    }
}

void ShaderWatcher::Quiesce() {
    {
        std::unique_lock lock(mutex_);
        stop_.store(true, std::memory_order_release);
        pending_reloads_.clear();
    }
    cv_.notify_all();
    if (debounce_thread_.joinable()) {
        debounce_thread_.join();
    }
}

void ShaderWatcher::OnFileChanged(const std::string& filename) {
    // All shader-manager access happens under mutex_ and is gated by stop_ so
    // that once Quiesce()/Stop() returns, no efsw callback can race the
    // destruction of the shader manager.
    std::unique_lock lock(mutex_);
    if (stop_.load(std::memory_order_acquire)) return;

    const auto ids = shaders_.FindAllBySlangFilename(filename);
    if (ids.empty()) return;

    const auto& slot = shaders_.GetSlot(ids.front());
    const std::string slang_path = slot.slang_path;
    LOGIFACE_LOG(debug, "ShaderWatcher: file changed: " + slang_path +
                 " (" + std::to_string(ids.size()) + " variant module(s))");

    pending_reloads_[slang_path] = std::chrono::steady_clock::now() + kReloadDebounce;
    cv_.notify_one();
}

void ShaderWatcher::DebounceLoop() {
    while (true) {
        std::vector<std::string> due;
        {
            std::unique_lock lock(mutex_);
            if (stop_.load(std::memory_order_acquire)) return;
            if (pending_reloads_.empty()) {
                cv_.wait(lock);
                continue;
            }
            const auto earliest = std::ranges::min_element(
                pending_reloads_, [](const auto& a, const auto& b) { return a.second < b.second; });
            cv_.wait_until(lock, earliest->second);
            if (stop_.load(std::memory_order_acquire)) return;
            const auto now = std::chrono::steady_clock::now();
            for (auto it = pending_reloads_.begin(); it != pending_reloads_.end();) {
                if (it->second <= now) {
                    due.push_back(it->first);
                    it = pending_reloads_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (const auto& path : due) {
            ReloadPath(path);
        }
    }
}

void ShaderWatcher::ReloadPath(const std::string& slang_path) {
    std::filesystem::file_time_type ftime;
    try {
        ftime = std::filesystem::last_write_time(slang_path);
    } catch (const std::filesystem::filesystem_error&) {
        return; // file may have been renamed away; a later event re-schedules
    }

    {
        std::unique_lock lock(mutex_);
        const auto it = last_compile_.find(slang_path);
        if (it != last_compile_.end() && ftime <= it->second) return; // duplicate event
        last_compile_[slang_path] = ftime;
    }
    // Reload every module built from this source (all entry-point variants).
    // RequestReloadByPath resolves to false when none match, so a stale path is
    // a no-op.
    (void)shaders_.RequestReloadByPath(slang_path);
}

} // namespace VulkanEngine::ShaderSystem
