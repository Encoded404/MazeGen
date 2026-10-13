module;

#include <logging/logging_macros.hpp>

#include <efsw/efsw.hpp>

module VulkanEngine.Text.FontWatcher;

import std;
import std.compat;

import logiface;

namespace VulkanEngine::Text {

namespace {

// The sfnt containers the loader can inspect. A changed file of any other type
// is not a font edit; ignoring it keeps a texture save in the same directory
// from triggering a pointless re-read.
constexpr std::array<std::string_view, 4> kFontExtensions{".ttf", ".otf", ".ttc", ".otc"};

struct WatcherListener final : efsw::FileWatchListener {
    FontWatcher& owner;
    std::string directory;

    WatcherListener(FontWatcher& owner_ref, std::string dir)
        : owner(owner_ref), directory(std::move(dir)) {}

    void handleFileAction(efsw::WatchID /*watchid*/, const std::string& /*dir*/,
                          const std::string& filename, efsw::Action action,
                          const std::string& /*oldFilename*/) override {
        if (action != efsw::Actions::Modified && action != efsw::Actions::Add) return;
        const bool font = std::ranges::any_of(kFontExtensions, [&filename](std::string_view ext) {
            return filename.size() >= ext.size() &&
                   std::string_view(filename).substr(filename.size() - ext.size()) == ext;
        });
        if (!font) return;
        owner.OnFileChanged(directory + "/" + filename);
    }
};

} // anonymous namespace

struct FontWatcher::Impl {
    // Destroyed in reverse declaration order: the watcher (joining its thread)
    // before the listeners.
    std::vector<std::unique_ptr<WatcherListener>> listeners;
    std::unique_ptr<efsw::FileWatcher> watcher;
};

FontWatcher::FontWatcher() = default;

FontWatcher::~FontWatcher() {
    Stop();
}

void FontWatcher::Start() {
    if (impl_) return;
    impl_ = std::make_unique<Impl>();
    impl_->watcher = std::make_unique<efsw::FileWatcher>();

    std::vector<std::string> known;
    {
        std::scoped_lock lock(mutex_);
        known.assign(registered_dirs_.begin(), registered_dirs_.end());
        registered_dirs_.clear();
    }
    // Re-register through AddDirectory so a watch id is created per directory.
    for (const auto& dir : known) {
        (void)AddDirectory(dir);
    }

    impl_->watcher->watch();
    stop_.store(false, std::memory_order_release);
}

bool FontWatcher::AddDirectory(const std::string& directory) {
    if (directory.empty()) {
        return false;
    }
    std::scoped_lock lock(mutex_);
    if (registered_dirs_.contains(directory)) {
        return true;
    }
    if (!impl_ || !impl_->watcher) {
        registered_dirs_.insert(directory);
        return true;
    }

    auto listener = std::make_unique<WatcherListener>(*this, directory);
    const auto watch_id =
        impl_->watcher->addWatch(directory, listener.get(), /*recursive=*/false);
    if (watch_id < 0) {
        LOGIFACE_LOG(warn, "FontWatcher: failed to watch " + directory);
        return false;
    }
    impl_->listeners.push_back(std::move(listener));
    registered_dirs_.insert(directory);
    LOGIFACE_LOG(info, "FontWatcher: watching " + directory);
    return true;
}

std::size_t FontWatcher::GetRegisteredDirectoryCount() const {
    std::scoped_lock lock(mutex_);
    return registered_dirs_.size();
}

void FontWatcher::Stop() {
    {
        std::scoped_lock lock(mutex_);
        stop_.store(true, std::memory_order_release);
        pending_.clear();
    }
    impl_.reset();
}

void FontWatcher::OnFileChanged(const std::string& full_path) {
    std::scoped_lock lock(mutex_);
    if (stop_.load(std::memory_order_acquire)) return;
    // Editors write several events per save; collapse a burst into one reload by
    // keeping the latest deadline per path.
    pending_[full_path] = std::chrono::steady_clock::now() + debounce_;
}

std::vector<std::string> FontWatcher::PollChanged() {
    std::vector<std::string> due;
    const auto now = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex_);
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (it->second <= now) {
            due.push_back(it->first);
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
    return due;
}

} // namespace VulkanEngine::Text
