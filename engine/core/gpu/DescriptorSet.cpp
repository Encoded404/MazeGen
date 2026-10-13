module;

#include <logging/logging_macros.hpp>

module VulkanEngine.GpuDescriptorSet;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.GpuBuffer;
import VulkanEngine.GpuTexture;
import VulkanShared.DeviceScope;

namespace VulkanEngine::GpuResources {

std::shared_ptr<DescriptorPool> DescriptorPool::Create(
    VulkanBackend::Vulkan::IVulkanBootstrap& backend,
    const DescriptorPoolConfig& config) {
    // clang-tidy false positive (clang-analyzer-core.uninitialized.Assign):
    // When constructing a shared_ptr to a type deriving from enable_shared_from_this,
    // the analyzer cannot model that _Sp_counted_base's constructor initializes
    // _M_weak_count to 0 before _M_weak_add_ref is called internally by the
    // shared_ptr constructor. It incorrectly flags the compound assignment in
    // __atomic_add_single as using uninitialized memory. This is a known analyzer
    // limitation that has been reported multiple times to the LLVM project.
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
    auto pool = std::shared_ptr<DescriptorPool>(new DescriptorPool());
    pool->Initialize(backend, config);
    return pool;
}

DescriptorPool::~DescriptorPool() {
    if (pool_ && backend_) {
        try {
            VulkanShared::DeviceScopeGuard guard;
            backend_->GetDevice().waitIdle();
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
    }
    pool_.reset();
    descriptor_set_layout_.reset();
    backend_ = nullptr;
}

void DescriptorPool::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend, const DescriptorPoolConfig& config) {
    LOGIFACE_LOG(trace, "entering DescriptorPool::Initialize");
    backend_ = &backend;
    config_ = config;

    const auto& device = backend.GetDevice();

    std::vector<vk::DescriptorPoolSize> pool_sizes;
    if (config.max_combined_image_samplers > 0) {
        pool_sizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, config.max_combined_image_samplers);
    }
    if (config.max_uniform_buffers > 0) {
        pool_sizes.emplace_back(vk::DescriptorType::eUniformBuffer, config.max_uniform_buffers);
    }
    if (config.max_storage_buffers > 0) {
        pool_sizes.emplace_back(vk::DescriptorType::eStorageBuffer, config.max_storage_buffers);
    }
    if (config.max_storage_images > 0) {
        pool_sizes.emplace_back(vk::DescriptorType::eStorageImage, config.max_storage_images);
    }
    if (config.max_sampled_images > 0) {
        pool_sizes.emplace_back(vk::DescriptorType::eSampledImage, config.max_sampled_images);
    }
    if (config.max_samplers > 0) {
        pool_sizes.emplace_back(vk::DescriptorType::eSampler, config.max_samplers);
    }

    if (pool_sizes.empty()) {
        throw std::runtime_error("DescriptorPoolConfig must specify at least one descriptor type");
    }

    const vk::DescriptorPoolCreateInfo pool_info(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, config.max_sets, pool_sizes);
    pool_ = std::make_unique<vk::raii::DescriptorPool>(device, pool_info);

    LOGIFACE_LOG(trace, "leaving DescriptorPool::Initialize successfully");
}

void DescriptorPool::SetDebugName(const vk::raii::Device& dev, const std::string& name) const {
    if (pool_) {
        VulkanBackend::Vulkan::SetVulkanObjectName(dev, *pool_, name);
    }
}

void DescriptorPool::FreeDescriptorSet(vk::DescriptorSet set) const {
    if (!pool_ || !backend_) {
        return;
    }
    static_cast<vk::Device>(backend_->GetDevice()).freeDescriptorSets(**pool_, set);
}

GpuDescriptorSet DescriptorPool::Allocate(vk::DescriptorSetLayout layout) {
    if (!pool_) {
        throw std::runtime_error("Descriptor pool not initialized");
    }

    const vk::DescriptorSetAllocateInfo alloc_info(**pool_, layout);
    
    auto sets = backend_->GetDevice().allocateDescriptorSets(alloc_info);

    return GpuDescriptorSet::Create(backend_, shared_from_this(), sets[0].release(), layout);
}

GpuDescriptorSet::GpuDescriptorSet(
    VulkanBackend::Vulkan::IVulkanBootstrap* backend,
    std::shared_ptr<DescriptorPool> pool,
    const vk::DescriptorSet set,
    const vk::DescriptorSetLayout layout)
    : backend_(backend)
    , pool_(std::move(pool))
    , descriptor_set_(set)
    , layout_(layout) {
}

GpuDescriptorSet::GpuDescriptorSet(GpuDescriptorSet&& other) noexcept
    : backend_(other.backend_)
    , pool_(std::move(other.pool_))
    , descriptor_set_(other.descriptor_set_)
    , layout_(other.layout_)
    , destroyed_(other.destroyed_) {
    other.descriptor_set_ = nullptr;
    other.layout_ = nullptr;
    other.destroyed_ = true;
}

GpuDescriptorSet& GpuDescriptorSet::operator=(GpuDescriptorSet&& other) noexcept {
    if (this != &other) {
        if (backend_) {
            try {
                VulkanShared::DeviceScopeGuard guard;
                backend_->GetDevice().waitIdle();
            } catch (...) { // NOLINT(bugprone-empty-catch)
            }
        }
        backend_ = other.backend_;
        pool_ = std::move(other.pool_);
        descriptor_set_ = other.descriptor_set_;
        layout_ = other.layout_;
        destroyed_ = other.destroyed_;
        other.descriptor_set_ = nullptr;
        other.layout_ = nullptr;
        other.destroyed_ = true;
    }
    return *this;
}

GpuDescriptorSet::~GpuDescriptorSet() {
    try {
        Destroy();
    } catch (const std::exception& err) { // NOLINT(bugprone-empty-catch)
        LOGIFACE_LOG(
            error,
            std::string("Exception type: ") +
            typeid(err).name() +
            ", message: " +
            err.what()
        );
    }
}

void GpuDescriptorSet::Destroy() {
    if (destroyed_ || !descriptor_set_ || !pool_) {
        return;
    }

    if (backend_) {
        try {
            VulkanShared::DeviceScopeGuard guard;
            backend_->GetDevice().waitIdle();
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
    }

    pool_->FreeDescriptorSet(descriptor_set_);
    descriptor_set_ = nullptr;
    layout_ = nullptr;
    destroyed_ = true;
}

void GpuDescriptorSet::SetDebugName(const vk::raii::Device& dev, const std::string& name) const {
    if (!descriptor_set_) return;

    VulkanBackend::Vulkan::SetVulkanObjectName(dev, descriptor_set_, vk::ObjectType::eDescriptorSet, name);
}

GpuDescriptorSet GpuDescriptorSet::Create(
    VulkanBackend::Vulkan::IVulkanBootstrap* backend,
    std::shared_ptr<DescriptorPool> pool,
    const vk::DescriptorSet set,
    const vk::DescriptorSetLayout layout) {
    return GpuDescriptorSet(backend, std::move(pool), set, layout);
}

void GpuDescriptorSet::UpdateBinding(std::uint32_t binding,
                                     const GpuTexture& texture,
                                     vk::ImageLayout layout) const {
    if (!backend_ || !descriptor_set_ || !texture.IsValid()) {
        return;
    }

    vk::DescriptorImageInfo image_info{};
    image_info.sampler = texture.GetSampler();
    image_info.imageView = texture.GetImageView();
    image_info.imageLayout = layout;

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = &image_info;

    backend_->GetDevice().updateDescriptorSets(write, nullptr);
}

void GpuDescriptorSet::UpdateBinding(std::uint32_t binding,
                                     const GpuBuffer& buffer,
                                     vk::DescriptorType type,
                                     std::uint64_t size,
                                     std::uint64_t offset) const {
    if (!backend_ || !descriptor_set_) {
        return;
    }

    const std::uint64_t buffer_size = size > 0 ? size : buffer.GetSize();

    const vk::DescriptorBufferInfo buffer_info(
        *buffer.GetBuffer(),
        offset,
        buffer_size);

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &buffer_info;

    backend_->GetDevice().updateDescriptorSets(write, nullptr);
}

void GpuDescriptorSet::UpdateBinding(std::uint32_t binding,
                                     std::uint32_t array_element,
                                     const GpuBuffer& buffer,
                                     vk::DescriptorType type,
                                     std::uint64_t size,
                                     std::uint64_t offset) const {
    if (!backend_ || !descriptor_set_) {
        return;
    }

    const std::uint64_t buffer_size = size > 0 ? size : buffer.GetSize();

    const vk::DescriptorBufferInfo buffer_info(
        *buffer.GetBuffer(),
        offset,
        buffer_size);

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = binding;
    write.dstArrayElement = array_element;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &buffer_info;

    backend_->GetDevice().updateDescriptorSets(write, nullptr);
}

void GpuDescriptorSet::UpdateBinding(std::uint32_t binding,
                                     vk::Buffer buffer,
                                     vk::DescriptorType type,
                                     std::uint64_t size,
                                     std::uint64_t offset) const {
    if (!backend_ || !descriptor_set_) {
        return;
    }

    const vk::DescriptorBufferInfo buffer_info(
        buffer,
        offset,
        size);

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &buffer_info;

    backend_->GetDevice().updateDescriptorSets(write, nullptr);
}

void GpuDescriptorSet::UpdateBinding(std::uint32_t binding,
                                     vk::ImageView image_view,
                                     vk::DescriptorType type,
                                     vk::ImageLayout layout) const {
    if (!backend_ || !descriptor_set_ || !image_view) {
        return;
    }

    vk::DescriptorImageInfo image_info{};
    image_info.imageView = image_view;
    image_info.imageLayout = layout;

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = &image_info;

    backend_->GetDevice().updateDescriptorSets(write, nullptr);
}

void GpuDescriptorSet::UpdateBinding(std::uint32_t binding,
                                     vk::Sampler sampler) const {
    if (!backend_ || !descriptor_set_ || !sampler) {
        return;
    }

    vk::DescriptorImageInfo image_info{};
    image_info.sampler = sampler;

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eSampler;
    write.pImageInfo = &image_info;

    backend_->GetDevice().updateDescriptorSets(write, nullptr);
}

} // namespace VulkanEngine::GpuResources
