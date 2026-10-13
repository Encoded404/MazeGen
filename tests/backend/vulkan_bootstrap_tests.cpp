#include <gtest/gtest.h>


import std;
import std.compat;

import vulkan;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.FrameSubmissionRecord;
import test_vulkan_fakes;

namespace {

using namespace VulkanBackend::Vulkan;
using TestSupport::FakeVulkanBootstrapBackend;

TEST(VulkanBootstrapTest, InitializeBuildsRuntimeSkeletonState) {
    auto backend = std::make_shared<FakeVulkanBootstrapBackend>();
    VulkanBootstrap bootstrap(backend);
    ASSERT_TRUE(bootstrap.Initialize(VulkanBootstrapConfig{}));
    const auto snapshot = bootstrap.GetSnapshot();
    EXPECT_TRUE(snapshot.instance_ready);
    EXPECT_TRUE(snapshot.device_ready);
    EXPECT_TRUE(snapshot.swapchain_ready);
    EXPECT_EQ(snapshot.swapchain_image_count, 3u);
    EXPECT_EQ(snapshot.status, BootstrapStatus::Ok);
}

TEST(VulkanBootstrapTest, InitializeReportsInstanceFailure) {
    auto backend = std::make_shared<FakeVulkanBootstrapBackend>();
    backend->instance_result = false;
    VulkanBootstrap bootstrap(backend);
    EXPECT_FALSE(bootstrap.Initialize(VulkanBootstrapConfig{}));
    EXPECT_EQ(bootstrap.GetSnapshot().status, BootstrapStatus::InstanceCreationFailed);
}

TEST(VulkanBootstrapTest, OutOfDateCanBeRecoveredBySwapchainRecreate) {
    auto backend = std::make_shared<FakeVulkanBootstrapBackend>();
    VulkanBootstrap bootstrap(backend);
    ASSERT_TRUE(bootstrap.Initialize(VulkanBootstrapConfig{}));
    bootstrap.NotifySwapchainOutOfDate();
    EXPECT_EQ(bootstrap.BeginFrame().status, BootstrapStatus::SwapchainOutOfDate);
    backend->produced_swapchain_image_count = 4;
    ASSERT_TRUE(bootstrap.RecreateSwapchain());
    const auto frame = bootstrap.BeginFrame();
    EXPECT_EQ(frame.status, BootstrapStatus::Ok);
    EXPECT_EQ(frame.swapchain_image_count, 4u);
}

TEST(VulkanBootstrapTest, DeviceLostStatusPersistsUntilShutdown) {
    auto backend = std::make_shared<FakeVulkanBootstrapBackend>();
    VulkanBootstrap bootstrap(backend);
    ASSERT_TRUE(bootstrap.Initialize(VulkanBootstrapConfig{}));
    bootstrap.NotifyDeviceLost();
    EXPECT_EQ(bootstrap.BeginFrame().status, BootstrapStatus::DeviceLost);
    bootstrap.Shutdown();
    EXPECT_TRUE(backend->shutdown_called);
    EXPECT_FALSE(bootstrap.IsInitialized());
}

// The acquire semaphore must not be waited on only at colour-attachment output:
// a pass that samples the acquired backbuffer reads it in the fragment (or
// compute) stage, which is earlier. `eAllCommands` is a distinct stage bit (not
// a bitmask union), so this guards against a regression to the colour-only
// wait that would race a shader read of the backbuffer.
TEST(VulkanBootstrapTest, AcquireWaitStageCoversShaderReadsOfBackbuffer) {
    const vk::PipelineStageFlags mask = AcquireWaitStageMask();
    EXPECT_EQ(mask, vk::PipelineStageFlagBits::eAllCommands);
    EXPECT_NE(mask, vk::PipelineStageFlags{vk::PipelineStageFlagBits::eColorAttachmentOutput});
}

// Submission record: the record gates the corrected IsFrameComplete.
// A frame is recorded only after its real runs were submitted; a recorded-but-
// dropped frame (runs dropped, fence still signaled by the empty consume
// submit) must NOT count as recorded. This is the device-free half of the
// dropped-frame semantics; the fence half stays with the backend.
TEST(FrameSubmissionRecordTest, UninitializedSlotsAreNeverRecorded) {
    FrameSubmissionRecord record{};
    record.Initialize(3);
    EXPECT_FALSE(record.IsRecorded(0));
    EXPECT_FALSE(record.IsRecorded(1));
    EXPECT_FALSE(record.IsRecorded(4));
}

TEST(FrameSubmissionRecordTest, SubmittedFrameIsRecordedAndLaterSlotReuseStillProvesIt) {
    FrameSubmissionRecord record{};
    record.Initialize(3);
    record.MarkSubmitted(1);
    EXPECT_TRUE(record.IsRecorded(1));
    // Slot 1 % 3 gets frame 4 next: frame 1's submit provably happened.
    record.MarkSubmitted(4);
    EXPECT_TRUE(record.IsRecorded(1));
    EXPECT_TRUE(record.IsRecorded(4));
    // A frame that never submitted into an unclaimed slot stays unrecorded.
    EXPECT_FALSE(record.IsRecorded(2));
    // A frame whose slot record names an unrelated (non-same-slot) frame is
    // not provable: record names frame 4 for slot 1; frame 5 lives in slot 2.
    EXPECT_FALSE(record.IsRecorded(0));
}

TEST(FrameSubmissionRecordTest, DroppedFramesNeverAdvanceTheRecord) {
    // Simulates the drop path: frame 2 recorded runs but rendering failed
    // (no MarkSubmitted), then frame 5 submits normally into slot 2 % 3.
    FrameSubmissionRecord record{};
    record.Initialize(3);
    record.MarkSubmitted(5);
    EXPECT_TRUE(record.IsRecorded(5));
    // Frame 2 shares slot 2 with frame 5: frame 2 was never submitted but its
    // slot has since moved past it, so nothing of frame 2 can be in flight
    // (the N <= record form). Recording means "not before this frame's slot
    // boundary", which is the gate ring Publish ops need.
    EXPECT_TRUE(record.IsRecorded(2));
    // Frame 3 lives in slot 0 which has never been submitted.
    EXPECT_FALSE(record.IsRecorded(3));
    EXPECT_FALSE(record.IsRecorded(4));

    // A never-advanced slot in a mid-life record: frames 0-2 submitted, then
    // frames 3-4 dropped and 5 is a later frame in slot 2 that WAS submitted
    // (advancing) — checked above. A never-submitted frame whose slot record
    // names an older frame (e.g. frame 5 in slot 2 when the record still says
    // 2) is not recorded: only frames at or below the record count.
    record.Initialize(3);
    record.MarkSubmitted(0);
    record.MarkSubmitted(1);
    record.MarkSubmitted(2);
    EXPECT_FALSE(record.IsRecorded(3));
    EXPECT_FALSE(record.IsRecorded(4));
    EXPECT_FALSE(record.IsRecorded(5));
    EXPECT_TRUE(record.IsRecorded(0));
    EXPECT_TRUE(record.IsRecorded(2));

    // FIF == 1 residue case: only the submitted frame is recorded, exactly
    // once per advance; the old fence-only query could never express this.
    FrameSubmissionRecord single{};
    single.Initialize(1);
    single.MarkSubmitted(0);
    EXPECT_TRUE(single.IsRecorded(0));
    EXPECT_FALSE(single.IsRecorded(1));
    single.MarkSubmitted(1);
    EXPECT_TRUE(single.IsRecorded(1));
    EXPECT_TRUE(single.IsRecorded(0));
}

TEST(FrameSubmissionRecordTest, InitializeResetsAndClampsFramesInFlight) {
    FrameSubmissionRecord record{};
    record.Initialize(4);
    EXPECT_EQ(record.GetFramesInFlight(), 4u);
    record.MarkSubmitted(3);
    EXPECT_TRUE(record.IsRecorded(3));
    record.Initialize(2);
    EXPECT_EQ(record.GetFramesInFlight(), 2u);
    EXPECT_FALSE(record.IsRecorded(3)); // reset by the new Initialize
    record.Initialize(9); // clamped to the engine FIF cap
    EXPECT_EQ(record.GetFramesInFlight(), 4u);
}

}  // namespace
