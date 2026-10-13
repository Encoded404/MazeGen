module;

module VulkanEngine.GpuBuffer;

import std;
import std.compat;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.MemoryUtils;
import VulkanBackend.Vulkan.ImageUtils;

namespace VulkanEngine::GpuResources {

namespace {

void CreateBufferResource(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                          std::uint64_t size,
                          vk::BufferUsageFlags usage,
                          vk::MemoryPropertyFlags properties,
                          std::unique_ptr<vk::raii::Buffer>& out_buffer,
                          std::unique_ptr<vk::raii::DeviceMemory>& out_memory) {
    vk::BufferCreateInfo const info({}, size, usage);
    out_buffer = std::make_unique<vk::raii::Buffer>(backend.GetDevice(), info);

    const vk::DeviceBufferMemoryRequirements mem_req_info{
        &info,
        nullptr
    };
    const vk::MemoryRequirements2 requirements = backend.GetDevice().getBufferMemoryRequirements(mem_req_info);
    const auto& reqs = requirements.memoryRequirements;

    vk::MemoryAllocateInfo const alloc(reqs.size,
        VulkanBackend::Vulkan::MemoryUtils::FindMemoryType(backend.GetPhysicalDevice(), reqs.memoryTypeBits, properties));
    out_memory = std::make_unique<vk::raii::DeviceMemory>(backend.GetDevice(), alloc);
    out_buffer->bindMemory(*out_memory, 0);
}

} // namespace

GpuBuffer GpuBuffer::Create(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                            std::uint64_t size,
                            vk::BufferUsageFlags usage,
                            vk::MemoryPropertyFlags properties,
                            const void* initial_data) {
    GpuBuffer buffer{};
    CreateBufferResource(backend, size, usage, properties, buffer.buffer_, buffer.memory_);
    buffer.size_ = size;

    if (initial_data) {
        buffer.Upload(initial_data, size);
    }

    return buffer;
}

void GpuBuffer::Upload(const void* data, std::uint64_t size) {
    void* mapped = memory_->mapMemory(0, size);
    std::memcpy(mapped, data, size);
    memory_->unmapMemory();
}

void GpuBuffer::UploadAt(const void* data, std::uint64_t size, std::uint64_t offset) {
    if (offset + size > size_) return;
    void* mapped = memory_->mapMemory(offset, size);
    std::memcpy(mapped, data, size);
    memory_->unmapMemory();
}

void* GpuBuffer::Map(uint64_t offset, std::uint64_t size) {
    return memory_->mapMemory(offset, size);
}

void GpuBuffer::Unmap() {
    memory_->unmapMemory();
}

} // namespace VulkanEngine::GpuResources
