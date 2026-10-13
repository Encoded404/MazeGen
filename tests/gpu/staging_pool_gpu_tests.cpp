#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;
import TestSupport.HeadlessVulkanBackend;
import VulkanEngine.GpuResources.StagingPool;
import VulkanEngine.GpuResources.DeviceBufferHeap;
import VulkanEngine.GpuBuffer;

namespace {

using VulkanEngine::GpuResources::StagingPool;
using VulkanEngine::GpuResources::StagingPoolConfig;

class StagingPoolGpuTest : public ::testing::Test {
protected:
    TestSupport::HeadlessVulkanBackend backend{};
    StagingPool pool{};

    void SetUp() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            GTEST_SKIP() << "no Vulkan device available";
        }
        ASSERT_TRUE(backend.Initialize());
        StagingPoolConfig config{};
        config.block_size = 1u << 20;         // 1 MiB blocks
        config.capacity_bytes = 4u << 20;     // 4 MiB ceiling
        ASSERT_TRUE(pool.Initialize(backend, config));
    }

    void TearDown() override {
        if (!TestSupport::IsGpuDeviceAvailable()) {
            return;
        }
        pool.Shutdown();
        backend.Shutdown();
    }
};

// An item-sized allocation is a mapped, host-visible range of the requested size.
TEST_F(StagingPoolGpuTest, AllocateReturnsMappedItemSizedRange) {
    auto alloc = pool.Allocate(4096);
    ASSERT_TRUE(alloc.has_value());
    EXPECT_EQ(alloc->size, 4096u);
    EXPECT_NE(alloc->mapped_ptr, nullptr);
    EXPECT_NE(alloc->buffer, vk::Buffer{nullptr});
    EXPECT_FALSE(alloc->dedicated);
    EXPECT_GE(pool.OutstandingBytes(), 4096u);
}

// The capacity ceiling provides back-pressure rather than unbounded growth.
TEST_F(StagingPoolGpuTest, CapacityProvidesBackPressure) {
    const auto capacity = pool.CapacityBytes();
    // Allocate in 1 MiB items up to the ceiling.
    std::vector<VulkanEngine::GpuResources::StagingAlloc> held;
    while (auto alloc = pool.Allocate(1u << 20)) {
        held.push_back(*alloc);
    }
    EXPECT_LE(pool.OutstandingBytes(), capacity);
    EXPECT_FALSE(pool.Allocate(1u << 20).has_value());
    // Releasing one makes room again.
    pool.Retire(held.front(), /*recording_frame=*/0);
    held.erase(held.begin());
    pool.BeginFrame(backend.GetFramesInFlight());
    EXPECT_TRUE(pool.Allocate(1u << 20).has_value());
}

// An item larger than the block size takes a dedicated buffer (never split).
TEST_F(StagingPoolGpuTest, OversizedItemUsesDedicatedBuffer) {
    auto alloc = pool.Allocate(2u << 20);
    ASSERT_TRUE(alloc.has_value());
    EXPECT_TRUE(alloc->dedicated);
    EXPECT_EQ(alloc->size, 2u << 20);
    EXPECT_NE(alloc->mapped_ptr, nullptr);
}

// The frame-gated path frees the range only at the drain, one FIF later.
TEST_F(StagingPoolGpuTest, RetireFreesAtTheFrameDrain) {
    auto alloc = pool.Allocate(4096);
    ASSERT_TRUE(alloc.has_value());
    const auto before = pool.OutstandingBytes();
    pool.Retire(*alloc, /*recording_frame=*/0);
    // Before the drain the bytes are still outstanding.
    EXPECT_EQ(pool.OutstandingBytes(), before);
    const std::uint32_t fif = backend.GetFramesInFlight();
    pool.BeginFrame(fif);
    EXPECT_EQ(pool.OutstandingBytes(), before - 4096u);
}

// The immediate path records, submits, waits and frees in one call.
TEST_F(StagingPoolGpuTest, FlushImmediateFreesAfterGpuCompletion) {
    auto dst = VulkanEngine::GpuResources::GpuBuffer::Create(
        backend, 4096, vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    ASSERT_TRUE(dst.IsValid());

    auto alloc = pool.Allocate(4096);
    ASSERT_TRUE(alloc.has_value());
    std::memset(alloc->mapped_ptr, 0x5A, 4096);
    pool.RecordBufferCopy(*alloc, *dst.GetBuffer(), 0);
    ASSERT_TRUE(pool.FlushImmediate());
    EXPECT_EQ(pool.OutstandingBytes(), 0u);
}

} // namespace
