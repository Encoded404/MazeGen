module;

export module VulkanEngine.Text.FontWatcher;

import std;

export namespace VulkanEngine::Text {

// Watches font source directories and reports changed font files on the main
// thread. Mirrors TextureWatcher's efsw/debounce model, but does not touch the
// text system itself: it collects debounced paths and the caller polls them, so
// the reload (which rebuilds faces and retires GPU pages) stays on the main
// thread where the frame pump can order it.
//
// The watcher is inert when the platform has no watcher backend; reload simply
// never fires, which is correct for a shipped build.
class FontWatcher {
public:
    FontWatcher();
    ~FontWatcher();

    FontWatcher(const FontWatcher&) = delete;
    FontWatcher& operator=(const FontWatcher&) = delete;

    void Start();
    void Stop();

    // Watches `directory` (non-recursive). May be called before Start(): the
    // directory is remembered and watched once Start() runs. Returns false only
    // when an already-started watcher rejects the path.
    bool AddDirectory(const std::string& directory);

    // Drains and returns the debounced set of changed file paths. Main-thread.
    [[nodiscard]] std::vector<std::string> PollChanged();

    [[nodiscard]] std::size_t GetRegisteredDirectoryCount() const;

    // Invoked from the efsw listener thread (internal use).
    void OnFileChanged(const std::string& full_path);

    // Invoked from the efsw listener thread on a missed-actions report:
    // efsw dropped queued events, so every candidate file in the watched
    // directory is scheduled; the debounce drops the unchanged ones.
    void OnMissedChanges(const std::string& directory);

    // Queues a changed path exactly as a filesystem event would. Public so a
    // caller (or a test) can trigger a reload without depending on platform
    // watcher timing.
    void NotifyChanged(const std::string& full_path) { OnFileChanged(full_path); }

    // Sets the debounce window. Tests use 0 to make a change immediately
    // pollable; production keeps the default.
    void SetDebounce(std::chrono::milliseconds window) { debounce_ = window; }

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
    mutable std::mutex mutex_;
    std::unordered_set<std::string> registered_dirs_;
    // Changed path -> debounce deadline. `PollChanged` returns and removes the
    // entries whose deadline has passed, so no timer thread is needed.
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_;
    std::atomic<bool> stop_{false};
    std::chrono::milliseconds debounce_{200};
};

} // namespace VulkanEngine::Text
