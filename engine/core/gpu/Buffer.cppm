module;

export module VulkanEngine.GpuBuffer;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;

export namespace VulkanEngine::GpuResources {

class GpuBuffer {
public:
    static GpuBuffer Create(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                            std::uint64_t size,
                            vk::BufferUsageFlags usage,
                            vk::MemoryPropertyFlags properties,
                            const void* initial_data = nullptr);

    void Upload(const void* data, std::uint64_t size);
    void UploadAt(const void* data, std::uint64_t size, std::uint64_t offset);
    void* Map(std::uint64_t offset, std::uint64_t size);
    void Unmap();

    [[nodiscard]] vk::raii::Buffer& GetBuffer() { return *buffer_; }
    [[nodiscard]] const vk::raii::Buffer& GetBuffer() const { return *buffer_; }
    [[nodiscard]] vk::raii::DeviceMemory& GetMemory() { return *memory_; }
    [[nodiscard]] const vk::raii::DeviceMemory& GetMemory() const { return *memory_; }
    [[nodiscard]] std::uint64_t GetSize() const { return size_; }
    [[nodiscard]] bool IsValid() const { return buffer_ != nullptr; }
    [[nodiscard]] vk::DeviceAddress GetDeviceAddress(const vk::raii::Device& dev) const {
        vk::BufferDeviceAddressInfo info{};
        info.sType = vk::StructureType::eBufferDeviceAddressInfo;
        info.buffer = static_cast<vk::Buffer>(**buffer_);
        return dev.getBufferAddress(info);
    }

private:
    std::unique_ptr<vk::raii::Buffer> buffer_;
    std::unique_ptr<vk::raii::DeviceMemory> memory_;
    std::uint64_t size_ = 0;
};

} // namespace VulkanEngine::GpuResources
