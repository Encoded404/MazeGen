module;

export module VulkanEngine.PhysicalCameraSystem;

import std;

import vulkan;

export import VulkanEngine.PhysicalCameraTypes;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanEngine.BindlessManager;
import VulkanEngine.GpuResources;
import VulkanEngine.ShaderManager;
import VulkanEngine.PipelineFactory;

export namespace VulkanEngine::PhysicalCamera {

// PhysicalCameraSystem: SDL3-based webcam capture + GPU compositing.
//
// - Capture runs on a per-stream worker thread (SDL camera functions are
//   thread-safe); frames are handed to the main thread latest-wins.
// - Frames stay in their native format (XRGB8888 / YUY2 / NV12) on the GPU;
//   no CPU pixel conversion is ever performed. YUV->RGB happens in the
//   composite shader at sample time.
// - Bindings draw a camera stream into a target texture (viewable in any
//   material/shader through the bindless array) with Vulkan-style
//   viewport/scissor placement, source crop, fit policy, flip and rotation.
//   One stream may bind to many targets; one target may receive many streams.
//
// All Vulkan work happens on the main thread (inside Execute, called from the
// render frame). The system is inert until a camera is opened.
class PhysicalCameraSystem {
public:
    PhysicalCameraSystem();
    ~PhysicalCameraSystem();

    PhysicalCameraSystem(const PhysicalCameraSystem&) = delete;
    PhysicalCameraSystem& operator=(const PhysicalCameraSystem&) = delete;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend,
                    VulkanEngine::BindlessManager::BindlessManager& bindless,
                    VulkanEngine::GpuResources::GpuImageHeap& image_heap,
                    VulkanEngine::GpuResources::SamplerCache& sampler_cache,
                    ShaderSystem::ShaderManager& shader_manager,
                    ShaderSystem::PipelineFactory& pipeline_factory,
                    ShaderSystem::ShaderId composite_vert_id,
                    ShaderSystem::ShaderId composite_frag_id);
    void Shutdown();

    void PollShaders(std::uint32_t frame_counter);

    // ── Capture ─────────────────────────────────────────────────────────
    std::vector<PhysicalCameraDeviceInfo> EnumerateDevices();
    PhysicalCameraHandle Open(std::uint32_t device_index, const PhysicalCameraOpenConfig& config = {});
    void Close(PhysicalCameraHandle camera);

    [[nodiscard]] PhysicalCameraState GetState(PhysicalCameraHandle camera) const;
    [[nodiscard]] const PhysicalCameraFormatInfo* GetFormat(PhysicalCameraHandle camera) const;
    [[nodiscard]] std::span<const std::uint32_t> GetNativeTextureSlots(PhysicalCameraHandle camera) const;

    // ── Composite targets ───────────────────────────────────────────────
    // Targets are plain bindless textures (eSampled | eColorAttachment) that
    // can be sampled by any material/shader. Destroyed automatically with the
    // system.
    PhysicalCameraTargetId CreateTarget(std::uint32_t width, std::uint32_t height);
    void DestroyTarget(PhysicalCameraTargetId target);
    [[nodiscard]] std::uint32_t GetTargetTextureSlot(PhysicalCameraTargetId target) const;

    // ── Bindings ────────────────────────────────────────────────────────
    PhysicalCameraBindingHandle Bind(PhysicalCameraHandle camera,
                                     PhysicalCameraTargetId target,
                                     const PhysicalCameraBindingConfig& config = {});
    void Unbind(PhysicalCameraBindingHandle binding);
    void UnbindAll(PhysicalCameraHandle camera);

    // ── Per-frame (main thread; frame command buffer already begun) ─────
    // Uploads newly captured frames to the GPU and re-composites all active
    // bindings. Called from the renderer before the scene graph executes so
    // scene passes sampling a target see this frame's content.
    void Execute(vk::CommandBuffer cmd, std::uint32_t frame_index);

    // SDL camera events (approval, denial, hotplug) — fed from the platform
    // event processor. `sdl_event` is a pointer to an SDL_Event.
    void ProcessSdlEvent(void* sdl_event);

    [[nodiscard]] bool IsInitialized() const;
    [[nodiscard]] std::size_t GetActiveStreamCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_{};
};

} // namespace VulkanEngine::PhysicalCamera
