module;

export module VulkanEngine.ShaderWatcher;

import std;
import VulkanEngine.ShaderManager;

export namespace VulkanEngine::ShaderSystem {

class ShaderWatcher {
public:
    explicit ShaderWatcher(ShaderManager& shaders);
    ~ShaderWatcher();

    ShaderWatcher(const ShaderWatcher&) = delete;
    ShaderWatcher& operator=(const ShaderWatcher&) = delete;

    void Start();
    void Stop();
    // Stops the watcher, then hands the whole object to a detached thread that
    // destroys it (fire-and-forget; never wait for completion afterwards).
    // The efsw FileWatcher death involves joining its internal thread, which
    // can block for its poll interval, so engine shutdown moves it off the
    // critical path. Stop()'s quiesce guarantees every efsw/debounce path has
    // stopped reaching the shader manager before the object moves. The whole
    // object is detached on purpose: the efsw listener captures `this` and must
    // outlive the join. Null is allowed and is a no-op.
    static void StopDetached(std::unique_ptr<ShaderWatcher> watcher);

    // ── Registration-time watching ──
    // A shader registered after Start() lives in a directory that was not
    // watched when Start() enumerated the manager. AddDirectory() watches it
    // immediately (or queues it if the watcher has not started), and Refresh()
    // re-scans the shader manager for directories registered since the last
    // Start()/Refresh(). Call Refresh() after registering application shaders.
    bool AddDirectory(const std::string& directory);
    void Refresh();
    [[nodiscard]] std::size_t GetRegisteredDirectoryCount() const;

    // Invoked from the efsw listener thread (internal use). Public only because
    // the listener type lives in the implementation file.
    void OnFileChanged(const std::string& filename);

private:
    struct Impl;

    ShaderManager& shaders_;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::filesystem::file_time_type> last_compile_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_reloads_;
    std::condition_variable cv_;
    std::thread debounce_thread_;
    std::atomic<bool> stop_{false};
    // Directories queued before Start() or already watched; guarded by mutex_.
    std::unordered_set<std::string> registered_dirs_;

    void DebounceLoop();
    void ReloadPath(const std::string& slang_path);
    // Sets the stop flag, clears pending reloads, wakes and joins the debounce
    // thread. After this returns, no efsw/debounce path can reach `shaders_`.
    void Quiesce();
};

} // namespace VulkanEngine::ShaderSystem
