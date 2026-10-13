#include <gtest/gtest.h>

import std;

import vulkan;
import VulkanEngine.GpuResources.FrameRing;

namespace {

using VulkanEngine::GpuResources::FrameRing;

struct TestOp {
    int value{0};
};

// Applies ops into a log; the gate honors a set of "submitted" frames.
// Timeline model: BeginFrame(N) runs at the START of frame N (draining the
// ring slot that frame N reuses); ops enqueued during frame N are drained at
// the start of frame N + FIF.
class LoggedRing {
public:
    explicit LoggedRing(std::uint32_t fif) {
        ring_.Initialize(fif);
    }

    void Enqueue(TestOp op, std::uint32_t frame, bool gated) {
        ring_.Enqueue(std::move(op), frame, gated);
    }

    void BeginFrame(std::uint32_t frame_index) {
        ring_.BeginFrame(
            frame_index,
            [this](const TestOp& op, std::uint32_t frame) {
                applied_.push_back({frame, op.value});
            },
            [this](std::uint32_t frame) {
                return submitted_.contains(frame);
            },
            [this](const TestOp& op, std::uint32_t frame) {
                dropped_.push_back({frame, op.value});
            });
    }

    void MarkSubmitted(std::uint32_t frame) { submitted_.insert(frame); }

    void Flush() {
        ring_.Flush([](TestOp& op, std::uint32_t) { op.value += 100; });
    }

    std::vector<std::pair<std::uint32_t, int>> applied_{};
    std::vector<std::pair<std::uint32_t, int>> dropped_{};
    std::set<std::uint32_t> submitted_{};

    FrameRing<TestOp> ring_;
};

TEST(FrameRingTest, UngatedOpsApplyOneCycleLater) {
    // Recording frame 3 enqueues an op; the drain happens at the start of
    // frame 3 + FIF, and never before.
    for (std::uint32_t fif = 1; fif <= 4; ++fif) {
        LoggedRing ring(fif);
        ring.BeginFrame(3); // frame start: drains nothing (ring empty)
        ring.Enqueue({.value = 7}, /*recording_frame=*/3, /*gated=*/false);
        for (std::uint32_t f = 4; f < 3 + fif; ++f) {
            ring.BeginFrame(f);
            EXPECT_TRUE(ring.applied_.empty()) << "fif=" << fif << " frame=" << f;
        }
        ring.BeginFrame(3 + fif);
        ASSERT_EQ(ring.applied_.size(), 1u) << "fif=" << fif;
        EXPECT_EQ(ring.applied_[0].second, 7);
    }
}

TEST(FrameRingTest, GatedOpAppliesWhenSubmitted) {
    LoggedRing ring(2);
    ring.BeginFrame(0);
    ring.MarkSubmitted(0);
    ring.Enqueue({.value = 1}, 0, /*gated=*/true);
    ring.BeginFrame(1); // next frame: not the drain slot yet
    EXPECT_TRUE(ring.applied_.empty());
    ring.BeginFrame(2); // start of frame 2 drains the ring of frame 0
    ASSERT_EQ(ring.applied_.size(), 1u);
    EXPECT_EQ(ring.applied_[0].second, 1);
    EXPECT_TRUE(ring.dropped_.empty());
}

TEST(FrameRingTest, GatedOpDroppedWhenFrameNeverSubmitted) {
    // A recorded-but-dropped frame never publishes.
    LoggedRing ring(2);
    ring.BeginFrame(1);
    ring.Enqueue({.value = 5}, 1, /*gated=*/true);
    ring.BeginFrame(3); // start of frame 3 drains the ring of frame 1
    EXPECT_TRUE(ring.applied_.empty());
    ASSERT_EQ(ring.dropped_.size(), 1u);
    EXPECT_EQ(ring.dropped_[0].second, 5);
}

TEST(FrameRingTest, DestructOnlyOpsIgnoreTheGate) {
    LoggedRing ring(3);
    ring.BeginFrame(2);
    ring.Enqueue({.value = 9}, 2, /*gated=*/false);
    ring.BeginFrame(5); // start of frame 5 drains the ring of frame 2
    EXPECT_TRUE(ring.dropped_.empty());
    ASSERT_EQ(ring.applied_.size(), 1u);
}

TEST(FrameRingTest, FlushAppliesEverything) {
    LoggedRing ring(2);
    ring.Enqueue({.value = 1}, 0, /*gated=*/true);
    ring.Enqueue({.value = 2}, 1, /*gated=*/true);
    ring.Flush();
    EXPECT_EQ(ring.ring_.PendingCount(), 0u);
}

}  // namespace
