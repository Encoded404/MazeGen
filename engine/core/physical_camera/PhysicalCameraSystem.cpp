module;

#include <SDL3/SDL_camera.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_surface.h>

#include <logging/logging_macros.hpp>

module VulkanEngine.PhysicalCameraSystem;

import std;
import std.compat;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanDebugUtils;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuResources;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;
import VulkanEngine.PhysicalCameraTypes;
import VulkanShared.DeviceScope;

namespace VulkanEngine::PhysicalCamera {

namespace {

constexpr std::uint32_t kStagingSlots = 3;
constexpr std::uint32_t kCompositePushSize = 64;

// Match the layout of CameraCompositePC in camera_composite.slang.
struct alignas(16) CameraCompositePush {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t format = 0;   // 0 = Xrgb8888, 1 = Yuy2, 2 = Nv12
    std::uint32_t src_slot0 = 0;
    std::uint32_t src_slot1 = 0;
    std::uint32_t flags = 0;    // bit0 flip_x, bit1 flip_y
    float src_x = 0.0f;
    float src_y = 0.0f;
    float src_w = 0.0f;
    float src_h = 0.0f;
    float src_full_w = 0.0f;
    float src_full_h = 0.0f;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct ImageSpec {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    vk::Format format = vk::Format::eUndefined;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t byte_size = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

[[nodiscard]] SDL_PixelFormat ToSdlFormat(PhysicalCameraPixelFormat format) {
    switch (format) {
        case PhysicalCameraPixelFormat::Xrgb8888: return SDL_PIXELFORMAT_XRGB8888;
        case PhysicalCameraPixelFormat::Yuy2: return SDL_PIXELFORMAT_YUY2;
        case PhysicalCameraPixelFormat::Nv12: return SDL_PIXELFORMAT_NV12;
        case PhysicalCameraPixelFormat::Unknown: break;
    }
    return SDL_PIXELFORMAT_UNKNOWN;
}

[[nodiscard]] bool MapSdlFormat(SDL_PixelFormat format, PhysicalCameraPixelFormat& out) {
    switch (format) {
        case SDL_PIXELFORMAT_XRGB8888:
        case SDL_PIXELFORMAT_ARGB8888:
            out = PhysicalCameraPixelFormat::Xrgb8888;
            return true;
        case SDL_PIXELFORMAT_YUY2:
            out = PhysicalCameraPixelFormat::Yuy2;
            return true;
        case SDL_PIXELFORMAT_NV12:
            out = PhysicalCameraPixelFormat::Nv12;
            return true;
        default:
            return false;
    }
}

// GPU image formats/dimensions for each supported pixel format.
[[nodiscard]] std::vector<ImageSpec> ImageSpecsFor(PhysicalCameraPixelFormat format,
                                                   std::uint32_t width, std::uint32_t height) {
    switch (format) {
        case PhysicalCameraPixelFormat::Xrgb8888:
            return { { vk::Format::eB8G8R8A8Unorm, width, height,
                       static_cast<std::uint64_t>(width) * height * 4 } };
        case PhysicalCameraPixelFormat::Yuy2:
            return { { vk::Format::eR8G8B8A8Unorm, width / 2, height,
                       static_cast<std::uint64_t>(width) * height * 2 } };
        case PhysicalCameraPixelFormat::Nv12:
            return { { vk::Format::eR8Unorm, width, height,
                       static_cast<std::uint64_t>(width) * height },
                     { vk::Format::eR8G8Unorm, width / 2, height / 2,
                       static_cast<std::uint64_t>(width) * height / 2 } };
        case PhysicalCameraPixelFormat::Unknown:
            break;
    }
    return {};
}

[[nodiscard]] std::uint64_t BytesPerFrame(PhysicalCameraPixelFormat format,
                                          std::uint32_t width, std::uint32_t height) {
    std::uint64_t total = 0;
    for (const auto& spec : ImageSpecsFor(format, width, height)) {
        total += spec.byte_size;
    }
    return total;
}

// Copies the SDL surface into a tight CPU buffer (row-by-row, honoring the
// driver's pitch). The layout matches the GPU images exactly.
void CopyFrameTight(SDL_Surface* surface, PhysicalCameraPixelFormat format,
                    std::uint32_t width, std::uint32_t height, std::uint8_t* dst) {
    const auto* src = static_cast<const std::uint8_t*>(surface->pixels);
    const std::uint32_t pitch = static_cast<std::uint32_t>(surface->pitch);
    if (pitch == 0) return;

    if (format == PhysicalCameraPixelFormat::Xrgb8888) {
        const std::uint32_t row_bytes = width * 4;
        for (std::uint32_t y = 0; y < height; ++y) {
            std::memcpy(dst + static_cast<std::uint64_t>(y) * row_bytes,
                        src + static_cast<std::uint64_t>(y) * pitch, row_bytes);
        }
    } else if (format == PhysicalCameraPixelFormat::Yuy2) {
        const std::uint32_t row_bytes = width * 2;
        for (std::uint32_t y = 0; y < height; ++y) {
            std::memcpy(dst + static_cast<std::uint64_t>(y) * row_bytes,
                        src + static_cast<std::uint64_t>(y) * pitch, row_bytes);
        }
    } else if (format == PhysicalCameraPixelFormat::Nv12) {
        const std::uint32_t y_row = width;
        for (std::uint32_t y = 0; y < height; ++y) {
            std::memcpy(dst + static_cast<std::uint64_t>(y) * y_row,
                        src + static_cast<std::uint64_t>(y) * pitch, y_row);
        }
        const auto* uv_src = src + static_cast<std::uint64_t>(pitch) * height;
        std::uint8_t* uv_dst = dst + static_cast<std::uint64_t>(width) * height;
        const std::uint32_t uv_row = width;
        for (std::uint32_t y = 0; y < height / 2; ++y) {
            std::memcpy(uv_dst + static_cast<std::uint64_t>(y) * uv_row,
                        uv_src + static_cast<std::uint64_t>(y) * pitch, uv_row);
        }
    }
}

void TransitionLayout(vk::CommandBuffer cmd, vk::Image image,
                      vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                      vk::PipelineStageFlags src_stage, vk::AccessFlags src_access,
                      vk::PipelineStageFlags dst_stage, vk::AccessFlags dst_access) {
    // Transitioning from UNDEFINED discards contents: no source access/stage.
    if (old_layout == vk::ImageLayout::eUndefined) {
        src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
        src_access = {};
    }
    vk::ImageMemoryBarrier barrier{};
    barrier.image = image;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.subresourceRange = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    cmd.pipelineBarrier(src_stage, dst_stage, {}, {}, {}, barrier);
}

} // namespace

// ── Internal state ──────────────────────────────────────────────────────

struct StagingSlot {
    VulkanEngine::GpuResources::GpuBuffer buffer{};
    void* mapping = nullptr; // NOLINT(misc-non-private-member-variables-in-classes)
};

struct CameraStream {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    SDL_Camera* sdl_camera = nullptr;
    SDL_CameraID camera_id = 0;
    PhysicalCameraPixelFormat gpu_format = PhysicalCameraPixelFormat::Unknown;
    PhysicalCameraFormatInfo format_info{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    PhysicalCameraState state = PhysicalCameraState::Closed;
    std::uint32_t generation = 0;

    // CPU latest frame (written by the capture worker thread)
    std::vector<std::uint8_t> cpu_frame{};
    std::uint64_t frame_bytes = 0;
    std::mutex cpu_mutex{};
    std::uint64_t frame_seq = 0;

    std::thread worker{};
    std::atomic<bool> worker_stop{false};
    bool worker_started = false;

    // GPU
    std::vector<std::uint32_t> source_slots{};
    // Generation-checked handles for the same slots; used to release them when
    // the stream closes so the bounded bindless allocator does not leak.
    std::vector<VulkanEngine::BindlessManager::TextureHandle> source_handles{};
    std::array<StagingSlot, kStagingSlots> staging{};
    vk::ImageLayout source_layout = vk::ImageLayout::eUndefined;
    std::uint64_t last_uploaded_seq = 0;
    bool has_content = false;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct CameraTarget {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t bindless_slot = 0;
    VulkanEngine::BindlessManager::TextureHandle handle{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    std::uint32_t generation = 0;
    bool valid = false;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct CameraBinding {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t camera_index = 0;
    std::uint32_t camera_generation = 0;
    std::uint32_t target_index = 0;
    std::uint32_t target_generation = 0;
    PhysicalCameraBindingConfig config{};
    std::uint32_t generation = 0;
    bool valid = false;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct PhysicalCameraSystem::Impl {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    VulkanBackend::Vulkan::IVulkanBootstrap* backend = nullptr;
    VulkanEngine::BindlessManager::BindlessManager* bindless = nullptr;
    VulkanEngine::GpuResources::GpuImageHeap* image_heap = nullptr;
    VulkanEngine::GpuResources::SamplerCache* sampler_cache = nullptr;
    ShaderSystem::ShaderManager* shader_manager = nullptr;
    ShaderSystem::PipelineFactory* pipeline_factory = nullptr;

    bool sdl_camera_inited = false;
    // Newest frame index seen by Execute; the recording frame for slot releases
    // issued outside the per-frame path (close/destroy).
    std::uint32_t last_frame = 0;

    std::vector<std::uint32_t> free_streams;
    std::vector<std::uint32_t> free_targets;
    std::vector<std::uint32_t> free_bindings;
    std::vector<std::unique_ptr<CameraStream>> streams;
    std::vector<CameraTarget> targets;
    std::vector<CameraBinding> bindings;

    // Composite pipeline (set 0 = bindless + push constant)
    ShaderSystem::ShaderId composite_vert_id = 0;
    ShaderSystem::ShaderId composite_frag_id = 0;
    std::unique_ptr<vk::raii::PipelineLayout> composite_pipeline_layout{};
    ShaderSystem::PipelineSlot composite_slot{};
    std::optional<ShaderSystem::GraphicsPipelineDesc> composite_desc{};

    // Helpers (defined below)
    static CameraStream* FindStream(Impl& impl, const PhysicalCameraHandle& handle);
    static CameraTarget* FindTarget(Impl& impl, const PhysicalCameraTargetId& id);
    static CameraBinding* FindBinding(Impl& impl, const PhysicalCameraBindingHandle& handle);
    // Releases every bindless slot a stream owns (frame-gated; the slot returns
    // to the free list when the decommit applies).
    static void ReleaseStreamSlots(Impl& impl, CameraStream& stream);
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

void PhysicalCameraSystem::Impl::ReleaseStreamSlots(Impl& impl, CameraStream& stream) {
    if (impl.bindless != nullptr) {
        for (const auto& handle : stream.source_handles) {
            impl.bindless->ReleaseSlot(handle, impl.last_frame);
        }
    }
    stream.source_handles.clear();
    stream.source_slots.clear();
}

CameraStream* PhysicalCameraSystem::Impl::FindStream(Impl& impl, const PhysicalCameraHandle& handle) {
    if (!handle.IsValid()) return nullptr;
    if (handle.index >= impl.streams.size()) return nullptr;
    auto& stream = impl.streams[handle.index];
    if (!stream) return nullptr;
    if (stream->generation != handle.generation) return nullptr;
    return stream.get();
}

CameraTarget* PhysicalCameraSystem::Impl::FindTarget(Impl& impl, const PhysicalCameraTargetId& id) {
    if (!id.IsValid()) return nullptr;
    if (id.index >= impl.targets.size()) return nullptr;
    auto& target = impl.targets[id.index];
    if (!target.valid) return nullptr;
    if (target.generation != id.generation) return nullptr;
    return &target;
}

CameraBinding* PhysicalCameraSystem::Impl::FindBinding(Impl& impl, const PhysicalCameraBindingHandle& handle) {
    if (!handle.IsValid()) return nullptr;
    if (handle.index >= impl.bindings.size()) return nullptr;
    auto& binding = impl.bindings[handle.index];
    if (!binding.valid) return nullptr;
    if (binding.generation != handle.generation) return nullptr;
    return &binding;
}

namespace {

void CameraWorkerMain(CameraStream& stream) {
    while (!stream.worker_stop.load(std::memory_order_relaxed)) {
        std::uint64_t timestamp = 0;
        SDL_Surface* frame = SDL_AcquireCameraFrame(stream.sdl_camera, &timestamp);
        if (frame == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(stream.cpu_mutex);
            CopyFrameTight(frame, stream.gpu_format, stream.width, stream.height,
                           stream.cpu_frame.data());
            ++stream.frame_seq;
        }
        SDL_ReleaseCameraFrame(stream.sdl_camera, frame);
    }
}

void StartWorker(CameraStream& stream) {
    if (stream.worker_started) return;
    if (!stream.worker_stop.load(std::memory_order_relaxed) && stream.sdl_camera != nullptr) {
        stream.worker_started = true;
        stream.worker = std::thread(CameraWorkerMain, std::ref(stream));
    }
}

void StopWorker(CameraStream& stream) {
    if (stream.worker.joinable()) {
        stream.worker_stop.store(true, std::memory_order_relaxed);
        stream.worker.join();
    }
}

} // namespace

// ── Public API ──────────────────────────────────────────────────────────

PhysicalCameraSystem::PhysicalCameraSystem() = default;

PhysicalCameraSystem::~PhysicalCameraSystem() {
    Shutdown();
}

bool PhysicalCameraSystem::Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                                      VulkanEngine::BindlessManager::BindlessManager& bindless,
                                      VulkanEngine::GpuResources::GpuImageHeap& image_heap,
                                      VulkanEngine::GpuResources::SamplerCache& sampler_cache,
                                      ShaderSystem::ShaderManager& shader_manager,
                                      ShaderSystem::PipelineFactory& pipeline_factory,
                                      ShaderSystem::ShaderId composite_vert_id,
                                      ShaderSystem::ShaderId composite_frag_id) {
    if (impl_) return true;

    auto impl = std::make_unique<Impl>();
    impl->backend = &backend;
    impl->bindless = &bindless;
    impl->image_heap = &image_heap;
    impl->sampler_cache = &sampler_cache;
    impl->shader_manager = &shader_manager;
    impl->pipeline_factory = &pipeline_factory;
    impl->composite_vert_id = composite_vert_id;
    impl->composite_frag_id = composite_frag_id;

    auto& dev = backend.GetDevice();

    vk::PushConstantRange pc_range{};
    pc_range.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    pc_range.offset = 0;
    pc_range.size = kCompositePushSize;

    std::array<vk::DescriptorSetLayout, 1> set_layouts{ *bindless.GetLayout() };
    vk::PipelineLayoutCreateInfo layout_info{};
    layout_info.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
    layout_info.pSetLayouts = set_layouts.data();
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &pc_range;
    impl->composite_pipeline_layout = std::make_unique<vk::raii::PipelineLayout>(dev, layout_info);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, *impl->composite_pipeline_layout,
                                               "physical-camera-composite-layout");

    vk::PipelineColorBlendAttachmentState composite_blend_attach{};
    composite_blend_attach.blendEnable = vk::True;
    composite_blend_attach.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    composite_blend_attach.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    composite_blend_attach.colorBlendOp = vk::BlendOp::eAdd;
    composite_blend_attach.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    composite_blend_attach.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    composite_blend_attach.alphaBlendOp = vk::BlendOp::eAdd;
    composite_blend_attach.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    ShaderSystem::GraphicsPipelineDesc desc{};
    desc.vertex_shader = composite_vert_id;
    desc.fragment_shader = composite_frag_id;
    desc.vertex_input = vk::PipelineVertexInputStateCreateInfo({}, 0, nullptr, 0, nullptr);
    desc.input_assembly = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList);
    desc.viewport = vk::PipelineViewportStateCreateInfo({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.polygonMode = vk::PolygonMode::eFill;
    rs.cullMode = vk::CullModeFlagBits::eNone;
    rs.frontFace = vk::FrontFace::eClockwise;
    rs.lineWidth = 1.0f;
    desc.rasterization = rs;
    desc.multisample = vk::PipelineMultisampleStateCreateInfo({}, vk::SampleCountFlagBits::e1);
    desc.depth_stencil = vk::PipelineDepthStencilStateCreateInfo({}, false, false, vk::CompareOp::eLess);
    desc.color_blend = vk::PipelineColorBlendStateCreateInfo({}, false, vk::LogicOp::eCopy, 0, nullptr);
    desc.color_blend_attachments = {composite_blend_attach};
    desc.dynamic_states = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    desc.layout = *impl->composite_pipeline_layout;
    desc.color_formats = { vk::Format::eR8G8B8A8Unorm };
    impl->composite_desc = desc;

    // Size the pipeline retire ring to the pipeline depth configured on the device.
    impl->composite_slot.SetFramesInFlight(backend.GetFramesInFlight());

    auto result = pipeline_factory.CreateGraphics(desc, shader_manager);
    if (!result.has_value()) {
        LOGIFACE_LOG(error, "PhysicalCameraSystem: failed to create composite pipeline");
        return false;
    }
    impl->composite_slot.Swap(std::move(result.value()), 0);
    VulkanBackend::Vulkan::SetVulkanObjectName(dev, impl->composite_slot.Get(),
                                               "physical-camera-composite-pipeline");

    impl_ = std::move(impl);
    LOGIFACE_LOG(info, "PhysicalCameraSystem initialized");
    return true;
}

void PhysicalCameraSystem::Shutdown() {
    if (!impl_) return;

    // The system owns every binding and target it handed out; close them all
    // here so apps don't have to unwind their camera resources manually.
    for (auto& stream : impl_->streams) {
        if (!stream) continue;
        StopWorker(*stream);
        if (stream->sdl_camera) {
            SDL_CloseCamera(stream->sdl_camera);
            stream->sdl_camera = nullptr;
        }
        stream->state = PhysicalCameraState::Closed;
    }

    if (impl_->backend) {
        try {
            VulkanShared::DeviceScopeGuard guard;
            impl_->backend->GetDevice().waitIdle();
        } catch (const std::exception& err) {
            LOGIFACE_LOG(warn, std::string("PhysicalCameraSystem: waitIdle during shutdown: ") + err.what());
        }
    }

    // The device is idle, so return every bindless slot synchronously instead of
    // leaving deferred decommits that would never drain after shutdown.
    if (impl_->bindless) {
        for (auto& target : impl_->targets) {
            if (target.handle.IsValid()) {
                impl_->bindless->ReleaseSlotImmediate(target.handle.slot);
                target.handle = {};
            }
            target.bindless_slot = 0;
        }
        for (auto& stream : impl_->streams) {
            if (!stream) continue;
            for (const auto& handle : stream->source_handles) {
                impl_->bindless->ReleaseSlotImmediate(handle.slot);
            }
            stream->source_handles.clear();
            stream->source_slots.clear();
        }
    }
    for (auto& target : impl_->targets) {
        target.valid = false;
        target.generation++;
    }
    for (auto& binding : impl_->bindings) {
        binding.valid = false;
        binding.generation++;
    }
    impl_->streams.clear();

    impl_->composite_desc.reset();
    impl_->composite_pipeline_layout.reset();

    if (impl_->sdl_camera_inited) {
        SDL_QuitSubSystem(SDL_INIT_CAMERA);
        impl_->sdl_camera_inited = false;
    }

    impl_.reset();
}

void PhysicalCameraSystem::PollShaders(std::uint32_t frame_counter) {
    if (!impl_ || !impl_->composite_desc) return;

    const auto rebuild = [this](ShaderSystem::ShaderManager& s)
        -> std::optional<ShaderSystem::PipelineProduct> {
        if (!impl_ || !impl_->composite_desc) return std::nullopt;
        auto result = impl_->pipeline_factory->CreateGraphics(*impl_->composite_desc, s);
        return result ? std::optional<ShaderSystem::PipelineProduct>(std::move(*result))
                      : std::nullopt;
    };

    impl_->composite_slot.RetireFrame(frame_counter);
    impl_->composite_slot.PollAndRebuild(*impl_->shader_manager,
                                         impl_->composite_vert_id,
                                         impl_->composite_frag_id,
                                         rebuild, frame_counter);
}

std::vector<PhysicalCameraDeviceInfo> PhysicalCameraSystem::EnumerateDevices() {
    std::vector<PhysicalCameraDeviceInfo> result;
    if (!impl_) return result;
    if (!impl_->sdl_camera_inited) {
        if (!SDL_InitSubSystem(SDL_INIT_CAMERA)) {
            LOGIFACE_LOG(error, std::string("PhysicalCameraSystem: SDL_InitSubSystem(CAMERA) failed: ") +
                                SDL_GetError());
            return result;
        }
        impl_->sdl_camera_inited = true;
    }

    int count = 0;
    SDL_CameraID* cameras = SDL_GetCameras(&count);
    if (cameras == nullptr) return result;

    result.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        PhysicalCameraDeviceInfo info{};
        info.index = static_cast<std::uint32_t>(i);
        const char* name = SDL_GetCameraName(cameras[i]);
        info.name = (name != nullptr && name[0] != '\0') ? std::string(name) : "webcam";
        switch (SDL_GetCameraPosition(cameras[i])) {
            case SDL_CAMERA_POSITION_FRONT_FACING:
                info.position = PhysicalCameraPosition::FrontFacing;
                break;
            case SDL_CAMERA_POSITION_BACK_FACING:
                info.position = PhysicalCameraPosition::BackFacing;
                break;
            default:
                info.position = PhysicalCameraPosition::Unknown;
                break;
        }
        result.push_back(std::move(info));
    }
    SDL_free(cameras);
    return result;
}

PhysicalCameraHandle PhysicalCameraSystem::Open(std::uint32_t device_index,
                                                const PhysicalCameraOpenConfig& config) {
    if (!impl_ || !impl_->backend) return {};

    if (!impl_->sdl_camera_inited) {
        if (!SDL_InitSubSystem(SDL_INIT_CAMERA)) {
            LOGIFACE_LOG(error, std::string("PhysicalCameraSystem: SDL_InitSubSystem(CAMERA) failed: ") +
                                SDL_GetError());
            return {};
        }
        impl_->sdl_camera_inited = true;
    }

    int count = 0;
    SDL_CameraID* cameras = SDL_GetCameras(&count);
    if (cameras == nullptr || device_index >= static_cast<std::uint32_t>(count)) {
        SDL_free(cameras);
        LOGIFACE_LOG(error, "PhysicalCameraSystem: device index out of range");
        return {};
    }
    const SDL_CameraID instance_id = cameras[device_index];
    SDL_free(cameras);

    for (const auto preferred : config.preferred_formats) {
        const auto sdl_format = ToSdlFormat(preferred);
        if (sdl_format == SDL_PIXELFORMAT_UNKNOWN) continue;

        std::uint32_t width = config.width != 0 ? config.width : 640;
        std::uint32_t height = config.height != 0 ? config.height : 480;

        SDL_CameraSpec request{};
        request.format = sdl_format;
        request.width = static_cast<int>(width);
        request.height = static_cast<int>(height);
        request.framerate_numerator = static_cast<int>(config.fps != 0 ? config.fps : 30);
        request.framerate_denominator = 1;

        SDL_Camera* camera = SDL_OpenCamera(instance_id, &request);
        if (camera == nullptr) {
            LOGIFACE_LOG(debug, std::string("PhysicalCameraSystem: SDL_OpenCamera failed: ") +
                                SDL_GetError());
            continue;
        }

        SDL_CameraSpec actual{};
        if (!SDL_GetCameraFormat(camera, &actual)) {
            SDL_CloseCamera(camera);
            continue;
        }
        PhysicalCameraPixelFormat gpu_format = PhysicalCameraPixelFormat::Unknown;
        if (!MapSdlFormat(actual.format, gpu_format)) {
            LOGIFACE_LOG(debug, "PhysicalCameraSystem: unsupported camera format, trying next");
            SDL_CloseCamera(camera);
            continue;
        }
        if (actual.width <= 0 || actual.height <= 0) {
            SDL_CloseCamera(camera);
            continue;
        }

        std::uint32_t stream_index = 0;
        if (!impl_->free_streams.empty()) {
            stream_index = impl_->free_streams.back();
            impl_->free_streams.pop_back();
        } else {
            stream_index = static_cast<std::uint32_t>(impl_->streams.size());
            impl_->streams.push_back(nullptr);
        }
        if (impl_->streams[stream_index]) {
            impl_->streams[stream_index]->generation++;
        } else {
            impl_->streams[stream_index] = std::make_unique<CameraStream>();
        }
        auto& stream = *impl_->streams[stream_index];

        stream.sdl_camera = camera;
        stream.camera_id = instance_id;
        stream.gpu_format = gpu_format;
        stream.format_info.format = gpu_format;
        stream.format_info.width = static_cast<std::uint32_t>(actual.width);
        stream.format_info.height = static_cast<std::uint32_t>(actual.height);
        stream.format_info.fps_numerator = static_cast<std::uint32_t>(actual.framerate_numerator);
        stream.format_info.fps_denominator = static_cast<std::uint32_t>(actual.framerate_denominator);
        stream.width = stream.format_info.width;
        stream.height = stream.format_info.height;
        stream.frame_bytes = BytesPerFrame(gpu_format, stream.width, stream.height);
        stream.cpu_frame.resize(static_cast<std::size_t>(stream.frame_bytes));
        stream.source_layout = vk::ImageLayout::eUndefined;
        stream.last_uploaded_seq = 0;
        stream.has_content = false;
        stream.worker_started = false;
        stream.worker_stop.store(false, std::memory_order_relaxed);
        stream.source_slots.clear();
        stream.source_handles.clear();

        // GPU source images + bindless slots
        const auto specs = ImageSpecsFor(gpu_format, stream.width, stream.height);
        bool gpu_ok = true;
        for (const auto& spec : specs) {
            auto texture = VulkanEngine::GpuResources::GpuTexture::CreateStream(
                *impl_->backend, *impl_->image_heap, spec.width, spec.height, spec.format,
                gpu_format == PhysicalCameraPixelFormat::Xrgb8888, impl_->sampler_cache);
            if (!texture.IsValid()) {
                gpu_ok = false;
                break;
            }
            const std::string name = "physical-camera-" + std::to_string(stream_index) +
                                     "-plane-" + std::to_string(stream.source_slots.size());
            const auto handle = impl_->bindless->AllocateTextureSlot(
                std::move(texture), VulkanEngine::ResourceId{name});
            if (!handle.has_value()) {
                gpu_ok = false;
                break;
            }
            stream.source_slots.push_back(handle->slot);
            stream.source_handles.push_back(*handle);
        }
        if (!gpu_ok) {
            SDL_CloseCamera(camera);
            stream.sdl_camera = nullptr;
            return {};
        }

        // Staging buffers (host-visible, mapped)
        bool staging_ok = true;
        for (auto& slot : stream.staging) {
            slot.buffer = VulkanEngine::GpuResources::GpuBuffer::Create(
                *impl_->backend, stream.frame_bytes,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            if (!slot.buffer.IsValid()) {
                staging_ok = false;
                break;
            }
            slot.mapping = slot.buffer.Map(0, stream.frame_bytes);
            if (slot.mapping == nullptr) {
                staging_ok = false;
                break;
            }
        }
        if (!staging_ok) {
            SDL_CloseCamera(camera);
            stream.sdl_camera = nullptr;
            return {};
        }

        // SDL 3.4: SDL_GetCameraPermissionState returns the
        // SDL_CameraPermissionState enum, not a bare int.
        const int permission = SDL_GetCameraPermissionState(camera);
        if (permission == SDL_CAMERA_PERMISSION_STATE_APPROVED) {
            stream.state = PhysicalCameraState::Active;
            StartWorker(stream);
        } else if (permission == SDL_CAMERA_PERMISSION_STATE_PENDING) {
            stream.state = PhysicalCameraState::AwaitingApproval;
        } else {
            stream.state = PhysicalCameraState::Error;
        }

        PhysicalCameraHandle handle{};
        handle.index = stream_index;
        handle.generation = stream.generation;

        LOGIFACE_LOG(info, std::format(
            "PhysicalCameraSystem: opened device {} as {}x{} (format {}, state {})",
            device_index, stream.width, stream.height,
            static_cast<std::uint32_t>(gpu_format), static_cast<std::uint32_t>(stream.state)));
        return handle;
    }

    return {};
}

void PhysicalCameraSystem::Close(PhysicalCameraHandle camera) {
    if (!impl_) return;
    auto* stream = PhysicalCameraSystem::Impl::FindStream(*impl_, camera);
    if (stream == nullptr) return;

    StopWorker(*stream);
    if (stream->sdl_camera) {
        SDL_CloseCamera(stream->sdl_camera);
        stream->sdl_camera = nullptr;
    }
    stream->state = PhysicalCameraState::Closed;
    stream->generation++;
    Impl::ReleaseStreamSlots(*impl_, *stream);
}

PhysicalCameraState PhysicalCameraSystem::GetState(PhysicalCameraHandle camera) const {
    if (!impl_) return PhysicalCameraState::Closed;
    const auto* stream = PhysicalCameraSystem::Impl::FindStream(*impl_, camera);
    return stream ? stream->state : PhysicalCameraState::Closed;
}

const PhysicalCameraFormatInfo* PhysicalCameraSystem::GetFormat(PhysicalCameraHandle camera) const {
    if (!impl_) return nullptr;
    const auto* stream = PhysicalCameraSystem::Impl::FindStream(*impl_, camera);
    return stream ? &stream->format_info : nullptr;
}

std::span<const std::uint32_t> PhysicalCameraSystem::GetNativeTextureSlots(PhysicalCameraHandle camera) const {
    static const std::span<const std::uint32_t> kEmpty{};
    if (!impl_) return kEmpty;
    const auto* stream = PhysicalCameraSystem::Impl::FindStream(*impl_, camera);
    return stream ? std::span<const std::uint32_t>(stream->source_slots) : kEmpty;
}

PhysicalCameraTargetId PhysicalCameraSystem::CreateTarget(std::uint32_t width, std::uint32_t height) {
    if (!impl_ || !impl_->backend) return {};
    if (width == 0 || height == 0) return {};

    auto texture = VulkanEngine::GpuResources::GpuTexture::CreateColorTarget(
        *impl_->backend, *impl_->image_heap, width, height,
        vk::Format::eR8G8B8A8Unorm, impl_->sampler_cache);
    if (!texture.IsValid()) return {};

    std::uint32_t target_index = 0;
    if (!impl_->free_targets.empty()) {
        target_index = impl_->free_targets.back();
        impl_->free_targets.pop_back();
    } else {
        target_index = static_cast<std::uint32_t>(impl_->targets.size());
        impl_->targets.emplace_back();
    }
    auto& target = impl_->targets[target_index];
    target.generation++;
    const auto handle = impl_->bindless->AllocateTextureSlot(
        std::move(texture), VulkanEngine::ResourceId{
            "physical-camera-target-" + std::to_string(target_index)});
    if (!handle.has_value()) {
        impl_->free_targets.push_back(target_index);
        return {};
    }
    target.bindless_slot = handle->slot;
    target.handle = *handle;
    target.width = width;
    target.height = height;
    target.layout = vk::ImageLayout::eUndefined;
    target.valid = true;

    PhysicalCameraTargetId id{};
    id.index = target_index;
    id.generation = target.generation;
    return id;
}

void PhysicalCameraSystem::DestroyTarget(PhysicalCameraTargetId target) {
    if (!impl_) return;
    auto* t = PhysicalCameraSystem::Impl::FindTarget(*impl_, target);
    if (t == nullptr) return;

    for (auto& binding : impl_->bindings) {
        if (binding.valid && binding.target_index == target.index &&
            binding.target_generation == target.generation) {
            binding.valid = false;
            binding.generation++;
            impl_->free_bindings.push_back(static_cast<std::uint32_t>(&binding - impl_->bindings.data()));
        }
    }
    if (impl_->bindless && t->handle.IsValid()) {
        impl_->bindless->ReleaseSlot(t->handle, impl_->last_frame);
        t->handle = {};
    }
    t->bindless_slot = 0;
    t->valid = false;
    t->generation++;
    impl_->free_targets.push_back(target.index);
}

std::uint32_t PhysicalCameraSystem::GetTargetTextureSlot(PhysicalCameraTargetId target) const {
    if (!impl_) return 0;
    const auto* t = PhysicalCameraSystem::Impl::FindTarget(*impl_, target);
    return t ? t->bindless_slot : 0;
}

PhysicalCameraBindingHandle PhysicalCameraSystem::Bind(PhysicalCameraHandle camera,
                                                       PhysicalCameraTargetId target,
                                                       const PhysicalCameraBindingConfig& config) {
    if (!impl_) return {};
    auto* stream = PhysicalCameraSystem::Impl::FindStream(*impl_, camera);
    auto* t = PhysicalCameraSystem::Impl::FindTarget(*impl_, target);
    if (stream == nullptr || t == nullptr) return {};

    std::uint32_t binding_index = 0;
    if (!impl_->free_bindings.empty()) {
        binding_index = impl_->free_bindings.back();
        impl_->free_bindings.pop_back();
    } else {
        binding_index = static_cast<std::uint32_t>(impl_->bindings.size());
        impl_->bindings.emplace_back();
    }
    auto& binding = impl_->bindings[binding_index];
    binding.generation++;
    binding.camera_index = camera.index;
    binding.camera_generation = camera.generation;
    binding.target_index = target.index;
    binding.target_generation = target.generation;
    binding.config = config;
    binding.valid = true;

    PhysicalCameraBindingHandle handle{};
    handle.index = binding_index;
    handle.generation = binding.generation;
    return handle;
}

void PhysicalCameraSystem::Unbind(PhysicalCameraBindingHandle binding) {
    if (!impl_) return;
    auto* b = PhysicalCameraSystem::Impl::FindBinding(*impl_, binding);
    if (b == nullptr) return;
    b->valid = false;
    b->generation++;
    impl_->free_bindings.push_back(binding.index);
}

void PhysicalCameraSystem::UnbindAll(PhysicalCameraHandle camera) {
    if (!impl_) return;
    for (auto& binding : impl_->bindings) {
        if (binding.valid && binding.camera_index == camera.index &&
            binding.camera_generation == camera.generation) {
            binding.valid = false;
            binding.generation++;
            impl_->free_bindings.push_back(static_cast<std::uint32_t>(&binding - impl_->bindings.data()));
        }
    }
}

void PhysicalCameraSystem::ProcessSdlEvent(void* sdl_event) {
    if (!impl_ || sdl_event == nullptr) return;
    const auto* event = static_cast<const SDL_Event*>(sdl_event);
    if (event->type != SDL_EVENT_CAMERA_DEVICE_APPROVED &&
        event->type != SDL_EVENT_CAMERA_DEVICE_DENIED &&
        event->type != SDL_EVENT_CAMERA_DEVICE_REMOVED) {
        return;
    }

    for (auto& stream : impl_->streams) {
        if (!stream || stream->sdl_camera == nullptr) continue;
        if (stream->camera_id != event->cdevice.which) continue;

        switch (event->type) {
            case SDL_EVENT_CAMERA_DEVICE_APPROVED:
                if (stream->state == PhysicalCameraState::AwaitingApproval) {
                    stream->state = PhysicalCameraState::Active;
                    StartWorker(*stream);
                }
                break;
            case SDL_EVENT_CAMERA_DEVICE_DENIED:
                stream->state = PhysicalCameraState::Error;
                break;
            case SDL_EVENT_CAMERA_DEVICE_REMOVED:
                if (stream->state == PhysicalCameraState::Active ||
                    stream->state == PhysicalCameraState::AwaitingApproval) {
                    stream->state = PhysicalCameraState::Lost;
                }
                break;
            default:
                break;
        }
    }
}

void PhysicalCameraSystem::Execute(vk::CommandBuffer cmd, std::uint32_t frame_index) {
    if (!impl_ || !impl_->backend) return;
    // Recording frame for slot releases issued off the per-frame path.
    impl_->last_frame = frame_index;

    const std::uint32_t staging_index = frame_index % kStagingSlots;

    // ── 1. Upload newly captured frames ─────────────────────────────────
    for (auto& stream_ptr : impl_->streams) {
        if (!stream_ptr) continue;
        auto& stream = *stream_ptr;

        // Resolve async permission/approval states.
        if (stream.state == PhysicalCameraState::AwaitingApproval && stream.sdl_camera) {
            const int permission = SDL_GetCameraPermissionState(stream.sdl_camera);
            if (permission == SDL_CAMERA_PERMISSION_STATE_APPROVED) {
                stream.state = PhysicalCameraState::Active;
                StartWorker(stream);
            } else if (permission == SDL_CAMERA_PERMISSION_STATE_DENIED) {
                stream.state = PhysicalCameraState::Error;
            }
        }
        if (stream.state != PhysicalCameraState::Active) continue;
        if (stream.source_slots.empty() || stream.frame_bytes == 0) continue;

        std::uint64_t seq = 0;
        {
            std::lock_guard<std::mutex> lock(stream.cpu_mutex);
            seq = stream.frame_seq;
        }
        if (seq == stream.last_uploaded_seq) continue;

        auto& staging = stream.staging[staging_index];
        if (staging.mapping == nullptr) continue;

        {
            std::lock_guard<std::mutex> lock(stream.cpu_mutex);
            std::memcpy(staging.mapping, stream.cpu_frame.data(), stream.frame_bytes);
        }

        const auto specs = ImageSpecsFor(stream.gpu_format, stream.width, stream.height);
        if (specs.empty()) continue;

        const vk::ImageLayout to_dst_layout = vk::ImageLayout::eTransferDstOptimal;
        if (stream.source_layout != to_dst_layout) {
            for (const auto slot : stream.source_slots) {
                const auto* gpu_tex = impl_->bindless->GetTexture(slot);
                if (gpu_tex == nullptr) continue;
                TransitionLayout(cmd, gpu_tex->GetImage(),
                                 stream.source_layout, to_dst_layout,
                                 vk::PipelineStageFlagBits::eFragmentShader,
                                 vk::AccessFlagBits::eShaderRead,
                                 vk::PipelineStageFlagBits::eTransfer,
                                 vk::AccessFlagBits::eTransferWrite);
            }
            stream.source_layout = to_dst_layout;
        }

        std::uint64_t offset = 0;
        for (std::size_t i = 0; i < specs.size(); ++i) {
            const auto& spec = specs[i];
            const auto* gpu_tex = impl_->bindless->GetTexture(stream.source_slots[i]);
            if (gpu_tex == nullptr) continue;
            vk::BufferImageCopy region{};
            region.bufferOffset = offset;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource = vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor, 0, 0, 1);
            region.imageOffset = vk::Offset3D{0, 0, 0};
            region.imageExtent = vk::Extent3D{spec.width, spec.height, 1};
            cmd.copyBufferToImage(static_cast<vk::Buffer>(*staging.buffer.GetBuffer()), gpu_tex->GetImage(),
                                  vk::ImageLayout::eTransferDstOptimal, region);
            offset += spec.byte_size;
        }

        const vk::ImageLayout to_read_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        for (const auto slot : stream.source_slots) {
            const auto* gpu_tex = impl_->bindless->GetTexture(slot);
            if (gpu_tex == nullptr) continue;
            TransitionLayout(cmd, gpu_tex->GetImage(),
                             vk::ImageLayout::eTransferDstOptimal, to_read_layout,
                             vk::PipelineStageFlagBits::eTransfer,
                             vk::AccessFlagBits::eTransferWrite,
                             vk::PipelineStageFlagBits::eFragmentShader,
                             vk::AccessFlagBits::eShaderRead);
        }
        stream.source_layout = to_read_layout;

        stream.last_uploaded_seq = seq;
        stream.has_content = true;
    }

    // ── 2. Composite bindings into targets ──────────────────────────────
    struct GroupEntry {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        CameraBinding* binding;
        CameraStream* stream;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };
    std::unordered_map<std::uint32_t, std::vector<GroupEntry>> groups;
    groups.reserve(impl_->targets.size());

    for (auto& binding : impl_->bindings) {
        if (!binding.valid) continue;
        if (binding.target_index >= impl_->targets.size()) continue;
        auto& target = impl_->targets[binding.target_index];
        if (!target.valid || target.generation != binding.target_generation) continue;
        if (binding.camera_index >= impl_->streams.size()) continue;
        auto& stream = impl_->streams[binding.camera_index];
        if (!stream || stream->generation != binding.camera_generation) continue;
        if (stream->state != PhysicalCameraState::Active || !stream->has_content) continue;
        if (stream->source_slots.empty()) continue;
        groups[binding.target_index].push_back({ &binding, stream.get() });
    }

    const auto bindless_set = impl_->bindless->GetDescriptorSet();
    const auto pipeline_layout = static_cast<vk::PipelineLayout>(*impl_->composite_pipeline_layout);

    for (auto& [target_index, entries] : groups) {
        auto& target = impl_->targets[target_index];
        if (!target.valid) continue;

        const auto* gpu_tex = impl_->bindless->GetTexture(target.bindless_slot);
        if (gpu_tex == nullptr) continue;

        if (target.layout != vk::ImageLayout::eColorAttachmentOptimal) {
            TransitionLayout(cmd, gpu_tex->GetImage(),
                             target.layout, vk::ImageLayout::eColorAttachmentOptimal,
                             vk::PipelineStageFlagBits::eFragmentShader,
                             vk::AccessFlagBits::eShaderRead,
                             vk::PipelineStageFlagBits::eColorAttachmentOutput,
                             vk::AccessFlagBits::eColorAttachmentWrite);
            target.layout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        bool clear_first = false;
        for (const auto& entry : entries) {
            clear_first = clear_first || entry.binding->config.clear_before;
        }

        vk::RenderingAttachmentInfo color_attach{};
        color_attach.imageView = gpu_tex->GetImageView();
        color_attach.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        color_attach.loadOp = clear_first ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
        color_attach.storeOp = vk::AttachmentStoreOp::eStore;
        if (clear_first) {
            color_attach.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        }

        vk::RenderingInfo render_info{};
        render_info.renderArea = vk::Rect2D(vk::Offset2D{0, 0},
                                            vk::Extent2D{target.width, target.height});
        render_info.layerCount = 1;
        render_info.colorAttachmentCount = 1;
        render_info.pColorAttachments = &color_attach;
        cmd.beginRendering(render_info);

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, impl_->composite_slot.Get());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout,
                               0, 1, &bindless_set, 0, nullptr);

        for (const auto& entry : entries) {
            const auto& binding = *entry.binding;
            const auto& stream = *entry.stream;
            const auto& cfg = binding.config;

            const float src_w = cfg.src_width != 0 ? static_cast<float>(cfg.src_width)
                                                   : static_cast<float>(stream.width);
            const float src_h = cfg.src_height != 0 ? static_cast<float>(cfg.src_height)
                                                    : static_cast<float>(stream.height);
            const float dst_w = cfg.dst_width != 0.0f ? cfg.dst_width : static_cast<float>(target.width);
            const float dst_h = cfg.dst_height != 0.0f ? cfg.dst_height : static_cast<float>(target.height);
            const float dst_x = cfg.dst_x;
            const float dst_y = cfg.dst_y;

            float vp_w = dst_w;
            float vp_h = dst_h;
            if (cfg.fit == FitMode::Contain || cfg.fit == FitMode::Crop) {
                const float scale = (cfg.fit == FitMode::Contain)
                    ? std::min(dst_w / src_w, dst_h / src_h)
                    : std::max(dst_w / src_w, dst_h / src_h);
                vp_w = src_w * scale;
                vp_h = src_h * scale;
            }

            vk::Viewport viewport{};
            viewport.x = dst_x + (dst_w - vp_w) * 0.5f;
            viewport.y = dst_y + (dst_h - vp_h) * 0.5f;
            viewport.width = vp_w;
            viewport.height = vp_h;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            cmd.setViewport(0, viewport);

            vk::Rect2D scissor{};
            scissor.offset = vk::Offset2D{static_cast<std::int32_t>(dst_x),
                                          static_cast<std::int32_t>(dst_y)};
            scissor.extent = vk::Extent2D{static_cast<std::uint32_t>(dst_w),
                                          static_cast<std::uint32_t>(dst_h)};
            cmd.setScissor(0, scissor);

            CameraCompositePush pc{};
            switch (stream.gpu_format) {
                case PhysicalCameraPixelFormat::Xrgb8888: pc.format = 0; break;
                case PhysicalCameraPixelFormat::Yuy2: pc.format = 1; break;
                case PhysicalCameraPixelFormat::Nv12: pc.format = 2; break;
                case PhysicalCameraPixelFormat::Unknown: continue;
            }
            pc.src_slot0 = stream.source_slots[0];
            pc.src_slot1 = stream.source_slots.size() > 1 ? stream.source_slots[1] : 0;
            pc.flags = (cfg.flip_x ? 1u : 0u) | (cfg.flip_y ? 2u : 0u);
            pc.src_x = static_cast<float>(cfg.src_x);
            pc.src_y = static_cast<float>(cfg.src_y);
            pc.src_w = src_w;
            pc.src_h = src_h;
            pc.src_full_w = static_cast<float>(stream.width);
            pc.src_full_h = static_cast<float>(stream.height);

            cmd.pushConstants(pipeline_layout,
                              vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                              0, static_cast<std::uint32_t>(sizeof(CameraCompositePush)), &pc);
            cmd.draw(3, 1, 0, 0);
        }

        cmd.endRendering();

        TransitionLayout(cmd, gpu_tex->GetImage(),
                         vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                         vk::PipelineStageFlagBits::eColorAttachmentOutput,
                         vk::AccessFlagBits::eColorAttachmentWrite,
                         vk::PipelineStageFlagBits::eFragmentShader,
                         vk::AccessFlagBits::eShaderRead);
        target.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
}

bool PhysicalCameraSystem::IsInitialized() const {
    return impl_ != nullptr;
}

std::size_t PhysicalCameraSystem::GetActiveStreamCount() const {
    if (!impl_) return 0;
    std::size_t count = 0;
    for (const auto& stream : impl_->streams) {
        if (stream && stream->state == PhysicalCameraState::Active) ++count;
    }
    return count;
}

} // namespace VulkanEngine::PhysicalCamera
