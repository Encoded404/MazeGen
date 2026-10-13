#include <gtest/gtest.h>

import std;

import vulkan;
import test_gpu;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Device-gated deterministic offscreen frame hash.
//
// This is the automated regression gate for GPU behavior that the validation
// layers cannot see: it clears an offscreen color image to a known value,
// copies it back through a real layout transition, and hashes the bytes. A
// future phase extends the same harness to render the engine's fixed scene;
// the device/queue/readback plumbing here is what makes that possible.
//
// Registered with the `gpu` ctest label. The default test preset excludes that
// label, so `ctest --preset debug` stays device-free. Run it with
// `ctest --preset debug-gpu`.
// ─────────────────────────────────────────────────────────────────────────────

constexpr std::uint32_t kSize = 64;
constexpr std::uint32_t kPixelCount = kSize * kSize;
constexpr std::uint32_t kBufferBytes = kPixelCount * 4;

// FNV-1a (64-bit) over the readback buffer's golden byte pattern.
constexpr std::uint64_t kGoldenHash = 0xf0c5ea0c5b91a325ULL;

std::uint64_t Fnv1a64(const std::uint8_t* data, std::size_t size) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint32_t FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                             std::uint32_t type_bits,
                             vk::MemoryPropertyFlags required) {
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if (((type_bits & (1u << i)) != 0u) &&
            (properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    throw std::runtime_error("no memory type satisfies the requested properties");
}

TEST(GpuOffscreenHashTest, DeterministicClearSurvivesReadback) {
    if (!TestSupport::IsGpuDeviceAvailable()) {
        GTEST_SKIP() << "no Vulkan device available";
    }

    const vk::raii::Context context;

    const vk::ApplicationInfo app_info("VulkanEngineV5-tests", 1, "VulkanEngineV5-tests", 1, vk::ApiVersion13);
    const vk::InstanceCreateInfo instance_info({}, &app_info);
    const vk::raii::Instance instance(context, instance_info);

    const auto physical_devices = instance.enumeratePhysicalDevices();
    ASSERT_FALSE(physical_devices.empty());
    const vk::raii::PhysicalDevice& physical_device = physical_devices.front();

    const auto queue_families = physical_device.getQueueFamilyProperties();
    std::uint32_t graphics_family = vk::QueueFamilyIgnored;
    for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
        if ((queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{}) {
            graphics_family = i;
            break;
        }
    }
    ASSERT_NE(graphics_family, vk::QueueFamilyIgnored);

    const float queue_priority = 1.0f;
    const vk::DeviceQueueCreateInfo queue_info({}, graphics_family, 1, &queue_priority);
    const vk::DeviceCreateInfo device_info({}, queue_info);
    const vk::raii::Device device(physical_device, device_info);
    const vk::raii::Queue queue = device.getQueue(graphics_family, 0);

    const vk::PhysicalDeviceMemoryProperties memory_properties = physical_device.getMemoryProperties();

    // Host-visible readback buffer, written by a transfer from the cleared image.
    const vk::BufferCreateInfo buffer_info(
        {},
        kBufferBytes,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive);
    const vk::raii::Buffer readback_buffer(device, buffer_info);
    const vk::MemoryRequirements buffer_requirements = readback_buffer.getMemoryRequirements();
    const vk::MemoryAllocateInfo buffer_alloc(
        buffer_requirements.size,
        FindMemoryType(memory_properties,
                       buffer_requirements.memoryTypeBits,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    const vk::raii::DeviceMemory readback_memory(device, buffer_alloc);
    readback_buffer.bindMemory(*readback_memory, 0);

    // Offscreen color target for the fixed "frame".
    const vk::ImageCreateInfo image_info(
        {},
        vk::ImageType::e2D,
        vk::Format::eR8G8B8A8Unorm,
        vk::Extent3D{kSize, kSize, 1},
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
        vk::ImageLayout::eUndefined);
    const vk::raii::Image image(device, image_info);
    const vk::MemoryRequirements image_requirements = image.getMemoryRequirements();
    const vk::MemoryAllocateInfo image_alloc(
        image_requirements.size,
        FindMemoryType(memory_properties,
                       image_requirements.memoryTypeBits,
                       vk::MemoryPropertyFlagBits::eDeviceLocal));
    const vk::raii::DeviceMemory image_memory(device, image_alloc);
    image.bindMemory(*image_memory, 0);

    const vk::raii::CommandPool command_pool(
        device, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, graphics_family));
    auto command_buffers = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(*command_pool, vk::CommandBufferLevel::ePrimary, 1));
    const vk::raii::CommandBuffer& command_buffer = command_buffers.front();

    const vk::ImageSubresourceRange full_range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

    command_buffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    const vk::ImageMemoryBarrier to_transfer_dst(
        {},
        vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        *image,
        full_range);
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   {},
                                   {},
                                   {},
                                   to_transfer_dst);

    // Float values chosen so the UNORM conversion is exact (no round-half ties):
    //   0 -> 0, 128/255 -> 128, 1 -> 255, 64/255 -> 64.
    const std::array<float, 4> clear_value{0.0f, 128.0f / 255.0f, 1.0f, 64.0f / 255.0f};
    command_buffer.clearColorImage(*image,
                                   vk::ImageLayout::eTransferDstOptimal,
                                   vk::ClearColorValue(clear_value),
                                   full_range);

    const vk::ImageMemoryBarrier to_transfer_src(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eTransferRead,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        *image,
        full_range);
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   {},
                                   {},
                                   {},
                                   to_transfer_src);

    const vk::BufferImageCopy copy_region(0,
                                          0,
                                          0,
                                          vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                                          vk::Offset3D{0, 0, 0},
                                          vk::Extent3D{kSize, kSize, 1});
    command_buffer.copyImageToBuffer(*image, vk::ImageLayout::eTransferSrcOptimal, *readback_buffer, copy_region);

    const vk::BufferMemoryBarrier to_host(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eHostRead,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        *readback_buffer,
        0,
        kBufferBytes);
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eHost,
                                   {},
                                   {},
                                   to_host,
                                   {});
    command_buffer.end();

    const vk::raii::Fence fence(device, vk::FenceCreateInfo());
    vk::SubmitInfo submit_info{};
    const std::array<vk::CommandBuffer, 1> submit_buffers{*command_buffer};
    submit_info.commandBufferCount = static_cast<std::uint32_t>(submit_buffers.size());
    submit_info.pCommandBuffers = submit_buffers.data();
    queue.submit(submit_info, *fence);

    ASSERT_EQ(device.waitForFences(*fence, vk::True, UINT64_MAX), vk::Result::eSuccess);

    std::vector<std::uint8_t> readback(kBufferBytes);
    const void* mapped = readback_memory.mapMemory(0, kBufferBytes);
    std::memcpy(readback.data(), mapped, kBufferBytes);
    readback_memory.unmapMemory();

    const std::array<std::uint8_t, 4> expected_pixel{0u, 128u, 255u, 64u};
    for (std::uint32_t pixel = 0; pixel < kPixelCount; ++pixel) {
        const std::size_t base = static_cast<std::size_t>(pixel) * 4;
        if (readback[base] != expected_pixel[0] || readback[base + 1] != expected_pixel[1] ||
            readback[base + 2] != expected_pixel[2] || readback[base + 3] != expected_pixel[3]) {
            FAIL() << "pixel " << pixel << " diverged from the cleared value: [" << +readback[base] << ", "
                   << +readback[base + 1] << ", " << +readback[base + 2] << ", " << +readback[base + 3] << "]";
        }
    }

    const std::uint64_t hash = Fnv1a64(readback.data(), readback.size());
    EXPECT_EQ(hash, kGoldenHash)
        << "offscreen frame hash changed; if the render is intentionally different, update kGoldenHash";
}

}  // namespace
