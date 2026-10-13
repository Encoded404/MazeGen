#include <gtest/gtest.h>

import std;

import VulkanEngine.Text.FontWatcher;

namespace {

using VulkanEngine::Text::FontWatcher;

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
TEST(FontWatcherTest, DebouncesChanges) {
    FontWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(20));
    watcher.NotifyChanged("/tmp/fonts/Lato-Regular.ttf");

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
    EXPECT_EQ(changed[0], "/tmp/fonts/Lato-Regular.ttf");
}

// Repeated notifications of the same path collapse to one pending reload and
// one poll result, even with no debounce window.
TEST(FontWatcherTest, DeduplicatesRepeatedChanges) {
    FontWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    watcher.NotifyChanged("/tmp/Lato-Regular.ttf");
    watcher.NotifyChanged("/tmp/Lato-Regular.ttf");
    watcher.NotifyChanged("/tmp/Lato-Regular.ttf");
    const auto changed = watcher.PollChanged();
    EXPECT_EQ(changed.size(), 1u);
}

// Polling drains: a second poll returns nothing until a new change arrives.
TEST(FontWatcherTest, PollDrains) {
    FontWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    watcher.NotifyChanged("/tmp/Lato-Bold.ttf");
    EXPECT_EQ(watcher.PollChanged().size(), 1u);
    EXPECT_TRUE(watcher.PollChanged().empty());
}

// A stopped watcher ignores new notifications.
TEST(FontWatcherTest, StopSuppressesChanges) {
    FontWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    watcher.Stop();
    watcher.NotifyChanged("/tmp/Lato-Black.ttf");
    EXPECT_TRUE(watcher.PollChanged().empty());
}

// Stop then Start puts the watcher back in service: the stopped window's
// notifications are dropped and a later one is polled normally.
TEST(FontWatcherTest, StopThenStartResumesNotifications) {
    FontWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    watcher.Stop();
    watcher.NotifyChanged("/tmp/dropped.ttf");
    EXPECT_TRUE(watcher.PollChanged().empty());

    watcher.Start();
    watcher.NotifyChanged("/tmp/resumed.ttf");
    const auto changed = watcher.PollChanged();
    ASSERT_EQ(changed.size(), 1u);
    EXPECT_EQ(changed[0], "/tmp/resumed.ttf");
    watcher.Stop();
}

// Directories are remembered before Start and de-duplicated.
TEST(FontWatcherTest, RegistersDirectoriesBeforeStart) {
    FontWatcher watcher;
    EXPECT_TRUE(watcher.AddDirectory("/tmp/fonts"));
    EXPECT_TRUE(watcher.AddDirectory("/tmp/fonts"));
    EXPECT_EQ(watcher.GetRegisteredDirectoryCount(), 1u);
    watcher.Stop();
}

// An empty directory is rejected rather than remembered.
TEST(FontWatcherTest, RejectsAnEmptyDirectory) {
    FontWatcher watcher;
    EXPECT_FALSE(watcher.AddDirectory(""));
    EXPECT_EQ(watcher.GetRegisteredDirectoryCount(), 0u);
}

}  // namespace
