#include <gtest/gtest.h>

import std;

import VulkanEngine.TextureWatcher;

namespace {

using VulkanEngine::TextureSystem::TextureWatcher;

// Poll `cond` every few ticks until it holds or `timeout` elapses, so tests
// never wait a fixed wall-clock duration without needing one.
static bool WaitFor(const std::function<bool()>& cond, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (cond()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return cond();
}

// A notified change is not immediately pollable while the debounce window is
// open, then becomes pollable once it elapses.
TEST(TextureWatcherTest, DebouncesChanges) {
    TextureWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(20));
    watcher.NotifyChanged("/tmp/textures/albedo.png");

    // The debounce property: right after the notify (within the first few ms,
    // well short of the 20ms window) the change must not be pollable yet.
    const auto immediate_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
    while (std::chrono::steady_clock::now() < immediate_deadline) {
        EXPECT_TRUE(watcher.PollChanged().empty());
    }

    // Once the window elapses the change appears and carries the right path.
    std::vector<std::string> changed;
    ASSERT_TRUE(WaitFor([&] {
        changed = watcher.PollChanged();
        return !changed.empty();
    }, std::chrono::milliseconds(500)));
    ASSERT_EQ(changed.size(), 1u);
    EXPECT_EQ(changed[0], "/tmp/textures/albedo.png");
}

// Repeated notifications of the same path collapse to one pending reload and
// one poll result.
TEST(TextureWatcherTest, DeduplicatesRepeatedChanges) {
    TextureWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    watcher.NotifyChanged("/tmp/a.png");
    watcher.NotifyChanged("/tmp/a.png");
    watcher.NotifyChanged("/tmp/a.png");
    const auto changed = watcher.PollChanged();
    EXPECT_EQ(changed.size(), 1u);
}

// Polling drains: a second poll returns nothing until a new change arrives.
TEST(TextureWatcherTest, PollDrains) {
    TextureWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    watcher.NotifyChanged("/tmp/b.png");
    EXPECT_EQ(watcher.PollChanged().size(), 1u);
    EXPECT_TRUE(watcher.PollChanged().empty());
}

// A stopped watcher ignores new notifications.
TEST(TextureWatcherTest, StopSuppressesChanges) {
    TextureWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    watcher.Stop();
    watcher.NotifyChanged("/tmp/c.png");
    EXPECT_TRUE(watcher.PollChanged().empty());
}

// Directories are remembered before Start and de-duplicated.
TEST(TextureWatcherTest, RegistersDirectoriesBeforeStart) {
    TextureWatcher watcher;
    EXPECT_TRUE(watcher.AddDirectory("/tmp/textures"));
    EXPECT_TRUE(watcher.AddDirectory("/tmp/textures"));
    EXPECT_EQ(watcher.GetRegisteredDirectoryCount(), 1u);
    watcher.Stop();
}

}  // namespace
