module;

export module VulkanEngine.GpuDescriptorSet;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;
export import VulkanBackend.Vulkan.VulkanDebugUtils;
export import VulkanEngine.GpuBuffer;
export import VulkanEngine.GpuTexture;

namespace VulkanEngine::GpuResources {
    export class GpuDescriptorSet;
}

export namespace VulkanEngine::GpuResources {

    struct DescriptorPoolConfig {
        std::uint32_t max_sets = 100;
        std::uint32_t max_combined_image_samplers = 100;
        std::uint32_t max_uniform_buffers = 10;
        std::uint32_t max_storage_buffers = 10;
        std::uint32_t max_storage_images = 10;
        std::uint32_t max_sampled_images = 0;
        std::uint32_t max_samplers = 0;
    };

    class DescriptorPool : public std::enable_shared_from_this<DescriptorPool> {
    public:
        static std::shared_ptr<DescriptorPool> Create(
            VulkanBackend::Vulkan::IVulkanBootstrap& backend,
            const DescriptorPoolConfig& config = {});

        ~DescriptorPool();

        DescriptorPool(const DescriptorPool&) = delete;
        DescriptorPool& operator=(const DescriptorPool&) = delete;
        DescriptorPool(DescriptorPool&&) = delete;
        DescriptorPool& operator=(DescriptorPool&&) = delete;

        GpuDescriptorSet Allocate(vk::DescriptorSetLayout layout);

        void SetDebugName(const vk::raii::Device& dev, const std::string& name) const;

        [[nodiscard]] vk::DescriptorSetLayout* GetLayout() { return descriptor_set_layout_ ? const_cast<vk::DescriptorSetLayout*>(&**descriptor_set_layout_) : nullptr; }
        [[nodiscard]] const vk::DescriptorSetLayout* GetLayout() const { return descriptor_set_layout_ ? &**descriptor_set_layout_ : nullptr; }
        [[nodiscard]] VulkanBackend::Vulkan::IVulkanBootstrap* GetBackend() { return backend_; }
        [[nodiscard]] const VulkanBackend::Vulkan::IVulkanBootstrap* GetBackend() const { return backend_; }

    private:
        friend class GpuDescriptorSet;

        DescriptorPool() = default;

        void Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend, const DescriptorPoolConfig& config);
        void FreeDescriptorSet(vk::DescriptorSet set) const;

        VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
        std::unique_ptr<vk::raii::DescriptorPool> pool_;
        std::unique_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout_;
        DescriptorPoolConfig config_{};
    };

    class GpuDescriptorSet {
    public:
        GpuDescriptorSet() = default;
        ~GpuDescriptorSet();

        GpuDescriptorSet(const GpuDescriptorSet&) = delete;
        GpuDescriptorSet& operator=(const GpuDescriptorSet&) = delete;
        GpuDescriptorSet(GpuDescriptorSet&& other) noexcept;
        GpuDescriptorSet& operator=(GpuDescriptorSet&& other) noexcept;

        void UpdateBinding(std::uint32_t binding,
                           const GpuTexture& texture,
                           vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal) const;

        void UpdateBinding(std::uint32_t binding,
                           const GpuBuffer& buffer,
                           vk::DescriptorType type,
                           std::uint64_t size = 0,
                           std::uint64_t offset = 0) const;

        void UpdateBinding(std::uint32_t binding,
                           std::uint32_t array_element,
                           const GpuBuffer& buffer,
                           vk::DescriptorType type,
                           std::uint64_t size = 0,
                           std::uint64_t offset = 0) const;

        void UpdateBinding(std::uint32_t binding,
                           vk::Buffer buffer,
                           vk::DescriptorType type,
                           std::uint64_t size,
                           std::uint64_t offset) const;

        void UpdateBinding(std::uint32_t binding,
                           vk::ImageView image_view,
                           vk::DescriptorType type,
                           vk::ImageLayout layout) const;

        void UpdateBinding(std::uint32_t binding,
                           vk::Sampler sampler) const;

        void SetDebugName(const vk::raii::Device& dev, const std::string& name) const;

        [[nodiscard]] vk::DescriptorSet GetHandle() const { return descriptor_set_; }
        [[nodiscard]] bool IsValid() const { return descriptor_set_ != nullptr; }

    private:
        friend class DescriptorPool;

        static GpuDescriptorSet Create(
            VulkanBackend::Vulkan::IVulkanBootstrap* backend,
            std::shared_ptr<DescriptorPool> pool,
            vk::DescriptorSet set,
            vk::DescriptorSetLayout layout);

        GpuDescriptorSet(
            VulkanBackend::Vulkan::IVulkanBootstrap* backend,
            std::shared_ptr<DescriptorPool> pool,
            vk::DescriptorSet set,
            vk::DescriptorSetLayout layout);

        void Destroy();

        VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
        std::shared_ptr<DescriptorPool> pool_;
        vk::DescriptorSet descriptor_set_ = nullptr;
        vk::DescriptorSetLayout layout_ = nullptr;
        bool destroyed_ = false;
    };

} // namespace VulkanEngine::GpuResources
