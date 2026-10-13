module;

#include <logging/logging_macros.hpp>

module VulkanBackend.Vulkan.VulkanBootstrapBackend;

import std;

import logiface;

import vulkan;

import VulkanBackend.Vulkan.VulkanInstance;
import VulkanBackend.Vulkan.VulkanDevice;
import VulkanBackend.Vulkan.VulkanSwapchain;
import VulkanBackend.Vulkan.FrameSubmissionRecord;
import VulkanShared.DeviceScope;
import VulkanShared.ScopedSection;

namespace VulkanBackend::Vulkan {

namespace {

VulkanShared::ScopedSection DebugSection(const std::string& name) {
    return VulkanShared::ScopedSection{name, [](const std::string& section, double ms) {
        LOGIFACE_LOG(debug, section + ": " + std::to_string(ms) + " ms");
    }};
}

class DefaultVulkanBootstrapBackend final : public IVulkanBootstrap {
public:
    [[nodiscard]] bool CreateInstance(const VulkanBootstrapConfig& config) override {
        config_ = config;
        instance_ = std::make_unique<VulkanInstance>();
        if (!instance_->Initialize(config)) {
            error_message_ = instance_->GetCapabilities().GetErrorMessage();
            requirements_unmet_ = instance_->GetCapabilities().HasUnmetRequirements();
            return false;
        }
        return true;
    }

    [[nodiscard]] bool SelectPhysicalDevice() override {
        if (!instance_) return false; // Ensure instance is initialized
        device_ = std::make_unique<VulkanDevice>();
        // Select the physical device (hard-floor check + one-time capability query).
        if (!device_->SelectPhysicalDevice(*instance_)) {
            error_message_ = device_->GetCapabilities().GetErrorMessage();
            requirements_unmet_ = device_->GetCapabilities().HasUnmetRequirements();
            return false;
        }
        return true;
    }

    [[nodiscard]] bool CreateLogicalDevice(std::uint32_t frames_in_flight) override {
        if (!device_ || !instance_) return false; // Ensure device and instance are initialized
        // Create the logical device and all associated resources using the correct frame count.
        if (!device_->CreateLogicalDeviceAndResources(frames_in_flight, config_)) {
            error_message_ = device_->GetCapabilities().GetErrorMessage();
            requirements_unmet_ = device_->GetCapabilities().HasUnmetRequirements();
            return false;
        }
        // Per-FIF submission record: a slot's initial state claims nothing;
        // the first entry for each slot comes from the first real submit into it.
        submission_record_.Initialize(frames_in_flight);
        LOGIFACE_LOG(info, device_->GetCapabilities().Summarize(instance_->GetCapabilities()));
        return true;
    }

    [[nodiscard]] bool CreateSwapchain(const std::uint32_t preferred_image_count, const PresentMode present_mode, std::uint32_t& out_image_count) override {
        if (device_ && device_->IsValid()) {
            device_->GetDevice().waitIdle();
        }
        swapchain_ = std::make_unique<VulkanSwapchain>();
        if (!swapchain_->Initialize(*instance_, *device_, preferred_image_count, present_mode)) {
            swapchain_->Shutdown();
            swapchain_.reset();
            render_finished_semaphores_.clear();
            return false;
        }
        render_finished_semaphores_.clear();
        const auto image_count = swapchain_->GetImageCount();
        render_finished_semaphores_.reserve(image_count);
        const vk::SemaphoreCreateInfo sem_info{};
        for (std::uint32_t i = 0; i < image_count; ++i) {
            render_finished_semaphores_.push_back(std::make_unique<vk::raii::Semaphore>(device_->GetDevice(), sem_info));
        }
        out_image_count = swapchain_->GetImageCount();
        return true;
    }

    [[nodiscard]] bool GetSwapchainExtent(std::uint32_t& out_width, std::uint32_t& out_height) const override {
        if (!swapchain_) return false;
        out_width = swapchain_->GetWidth();
        out_height = swapchain_->GetHeight();
        return out_width > 0 && out_height > 0;
    }

    [[nodiscard]] const vk::raii::Instance& GetInstance() const override { return instance_->GetInstance(); }
    [[nodiscard]] const vk::raii::PhysicalDevice& GetPhysicalDevice() const override { return device_->GetPhysicalDevice(); }
    [[nodiscard]] const vk::raii::Device& GetDevice() const override { return device_->GetDevice(); }
    [[nodiscard]] const vk::raii::Queue& GetGraphicsQueue() const override { return device_->GetGraphicsQueue(); }
    [[nodiscard]] std::uint32_t GetGraphicsQueueFamily() const override { return device_->GetGraphicsQueueFamily(); }
    [[nodiscard]] const vk::raii::CommandPool& GetCommandPool() const override { return device_->GetCommandPool(); }
    [[nodiscard]] bool HasAsyncCompute() const override { return device_->HasAsyncCompute(); }
    [[nodiscard]] const vk::raii::Queue& GetComputeQueue() const override { return device_->GetComputeQueue(); }
    [[nodiscard]] std::uint32_t GetComputeQueueFamily() const override { return device_->GetComputeQueueFamily(); }
    [[nodiscard]] const vk::raii::CommandPool& GetComputeCommandPool() const override { return device_->GetComputeCommandPool(); }
    [[nodiscard]] vk::raii::CommandBuffer& GetComputeCommandBuffer(std::uint32_t frame_idx) override { return device_->GetComputeCommandBuffer(frame_idx); }
    [[nodiscard]] std::span<const std::uint32_t> GetQueueFamilies() const override { return device_->GetQueueFamilies(); }
    [[nodiscard]] vk::raii::CommandBuffer& GetRunCommandBuffer(bool compute, std::uint32_t frame_idx,
                                                               std::uint32_t run_slot) override {
        return device_->GetRunCommandBuffer(compute, frame_idx, run_slot);
    }
    [[nodiscard]] const vk::raii::Semaphore& GetRunSemaphore(std::uint32_t frame_idx,
                                                             std::uint32_t run_slot) const override {
        return device_->GetRunSemaphore(frame_idx, run_slot);
    }
    void SetFrameRuns(std::span<const QueueRunSubmit> runs) override {
        frame_runs_.assign(runs.begin(), runs.end());
    }

    [[nodiscard]] const VulkanCapabilities& GetCapabilities() const override { return device_->GetCapabilities(); }
    [[nodiscard]] const std::string& GetErrorMessage() const override { return error_message_; }
    [[nodiscard]] bool HasUnmetRequirements() const override { return requirements_unmet_; }

    [[nodiscard]] const vk::raii::Fence& GetInFlightFence(std::uint32_t frame_idx) const override { return device_->GetInFlightFence(frame_idx); }
    [[nodiscard]] const vk::raii::Semaphore& GetImageAvailableSemaphore(std::uint32_t frame_idx) const override { return device_->GetImageAvailableSemaphore(frame_idx); }
    [[nodiscard]] const vk::raii::Semaphore& GetRenderFinishedSemaphore(std::uint32_t image_idx) const override { return *render_finished_semaphores_.at(image_idx); }
    [[nodiscard]] vk::raii::CommandBuffer& GetCommandBuffer(std::uint32_t frame_idx) override { return device_->GetCommandBuffer(frame_idx); }

    [[nodiscard]] std::uint32_t GetFramesInFlight() const override { return device_->GetFramesInFlight(); }
    [[nodiscard]] std::uint32_t GetRunSlotsPerFrame() const override {
        return VulkanBackend::Vulkan::VulkanDevice::kRunSlotsPerFrame;
    }

    [[nodiscard]] const vk::raii::SwapchainKHR& GetSwapchain() const override { return swapchain_->GetSwapchain(); }
    [[nodiscard]] const std::vector<vk::Image>& GetSwapchainImages() const override { return swapchain_->GetImages(); }
    [[nodiscard]] const std::vector<vk::raii::ImageView>& GetSwapchainImageViews() const override { return swapchain_->GetImageViews(); }
    [[nodiscard]] std::vector<bool>& GetSwapchainImageInitializedFlags() override { return swapchain_->GetImageInitializedFlags(); }
    [[nodiscard]] const vk::SurfaceFormatKHR& GetSurfaceFormat() const override { return swapchain_->GetSurfaceFormat(); }
    [[nodiscard]] vk::Format GetDepthFormat() const override { return swapchain_->GetDepthFormat(); }
    [[nodiscard]] const vk::raii::ImageView& GetDepthImageView(std::uint32_t image_index) const override { return swapchain_->GetDepthImageView(image_index); }
    [[nodiscard]] const vk::raii::Image& GetDepthImage(std::uint32_t image_index) const override { return swapchain_->GetDepthImage(image_index); }

    [[nodiscard]] bool AcquireNextImage(std::uint32_t frame_idx, std::uint32_t& out_image_index) override {
        LOGIFACE_LOG(trace, "entering AcquireNextImage with frame index " + std::to_string(frame_idx));
        if (!device_ || !swapchain_) return false;

        const vk::raii::Device& vk_device = device_->GetDevice();
        const vk::raii::Fence& vk_in_flight_fence = device_->GetInFlightFence(frame_idx);
        const vk::raii::Semaphore& vk_image_available_semaphore = device_->GetImageAvailableSemaphore(frame_idx);

        constexpr std::uint64_t timeout = 1000000000; // 1 second

        // Wait for previous work to finish
        if (vk_device.waitForFences({*vk_in_flight_fence}, vk::True, timeout) != vk::Result::eSuccess)
        {
            LOGIFACE_LOG(error, "Failed to wait for in-flight fence during AcquireNextImage.");
            return false;
        }

        vk::AcquireNextImageInfoKHR acquire_info{};
        acquire_info.swapchain = *swapchain_->GetSwapchain();
        acquire_info.timeout = timeout;
        acquire_info.semaphore = *vk_image_available_semaphore;
        acquire_info.deviceMask = 1;

        try {
            auto [Result, image_index] = vk_device.acquireNextImage2KHR(acquire_info);
            if (Result != vk::Result::eSuccess && Result != vk::Result::eSuboptimalKHR) {
                return false;
            }
            out_image_index = image_index;
        } catch (...) {
            LOGIFACE_LOG(error, "Failed to acquire next image during AcquireNextImage.");
            return false;
        }

        current_image_index_ = out_image_index;
        LOGIFACE_LOG(trace, "leaving AcquireNextImage successfully. image index is: " + std::to_string(out_image_index));
        return true;
    }

    [[nodiscard]] bool SubmitFrame(std::uint32_t frame_idx, std::uint32_t image_index, bool rendering_succeeded) override {
        LOGIFACE_LOG(trace, "entering SubmitFrame with frame index " + std::to_string(frame_idx) +
                     " and image index " + std::to_string(image_index) + " and rendering succeeded " +
                     std::to_string(rendering_succeeded) + ".");

        // Take ownership immediately so a frame that bails out before submitting
        // cannot leave stale queue runs to be submitted by a later frame.
        std::vector<IVulkanBootstrap::QueueRunSubmit> runs = std::move(frame_runs_);
        frame_runs_.clear();
        if (!device_ || !swapchain_) return false;

        const vk::raii::Device& vk_device = device_->GetDevice();
        const vk::raii::Queue& vk_graphics_queue = device_->GetGraphicsQueue();
        const vk::raii::CommandBuffer& vk_command_buffer = device_->GetCommandBuffer(frame_idx);
        const vk::raii::Semaphore& vk_image_available_semaphore = device_->GetImageAvailableSemaphore(frame_idx);
        const vk::raii::Semaphore& vk_render_finished_semaphore = rendering_succeeded
            ? GetRenderFinishedSemaphore(image_index)
            : vk_image_available_semaphore; // unused when no command buffer is submitted
        const vk::raii::Fence& vk_in_flight_fence = device_->GetInFlightFence(frame_idx);

        // Reset the fence ONLY when we are about to submit work.
        vk_device.resetFences({*vk_in_flight_fence});

        // Multi-queue path: one submission per recorded run, ordered by binary
        // semaphores at cross-queue boundaries (same-queue runs are ordered by
        // queue submission order). An unsuccessful frame submits no runs, just
        // like the single-queue path below consumes the acquire semaphore.
        if (!runs.empty() && rendering_succeeded) {
            const std::uint32_t run_count = static_cast<std::uint32_t>(runs.size());
            const vk::raii::Queue& vk_compute_queue = device_->GetComputeQueue();

            std::vector<vk::Semaphore> wait_semaphores(run_count, nullptr);
            std::vector<vk::Semaphore> signal_semaphores(run_count, nullptr);
            std::vector<vk::PipelineStageFlags> wait_stages(run_count, AcquireWaitStageMask());
            for (std::uint32_t i = 0; i < run_count; ++i) {
                if (i == 0) {
                    wait_semaphores[i] = *vk_image_available_semaphore;
                } else if (runs[i].compute != runs[i - 1].compute) {
                    wait_semaphores[i] = *device_->GetRunSemaphore(frame_idx, i);
                }
                if (i + 1 == run_count) {
                    signal_semaphores[i] = *vk_render_finished_semaphore;
                } else if (runs[i + 1].compute != runs[i].compute) {
                    signal_semaphores[i] = *device_->GetRunSemaphore(frame_idx, i + 1);
                }
            }

            for (std::uint32_t i = 0; i < run_count; ++i) {
                vk::SubmitInfo submit_info{};
                if (wait_semaphores[i] != nullptr) {
                    submit_info.waitSemaphoreCount = 1;
                    submit_info.pWaitSemaphores = &wait_semaphores[i];
                    submit_info.pWaitDstStageMask = &wait_stages[i];
                }
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &runs[i].command_buffer;
                if (signal_semaphores[i] != nullptr) {
                    submit_info.signalSemaphoreCount = 1;
                    submit_info.pSignalSemaphores = &signal_semaphores[i];
                }
                const bool is_last = (i + 1 == run_count);
                const vk::raii::Queue& queue = runs[i].compute ? vk_compute_queue : vk_graphics_queue;
                queue.submit({submit_info}, is_last ? *vk_in_flight_fence : nullptr);
            }
            // Real runs were submitted into this FIF slot; the record advances
            // only here, never for a frame whose runs were dropped.
            submission_record_.MarkSubmitted(frame_idx);
            return true;
        }
        if (!runs.empty()) {
            LOGIFACE_LOG(debug, "SubmitFrame: dropping " + std::to_string(runs.size()) +
                                    " recorded queue runs for an unsuccessful frame");
        }

        const vk::PipelineStageFlags wait_stage = AcquireWaitStageMask();
        vk::SubmitInfo submit_info{};
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &*vk_image_available_semaphore;
        submit_info.pWaitDstStageMask = &wait_stage;

        if (rendering_succeeded) {
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &*vk_command_buffer;
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &*vk_render_finished_semaphore;
            // Real work submitted: record this frame for slot frame_idx % FIF.
            submission_record_.MarkSubmitted(frame_idx);
        } else {
            // If rendering didn't succeed, we still need to consume the
            // image_available_semaphore but we don't signal render_finished_semaphore.
            submit_info.commandBufferCount = 0;
            submit_info.pCommandBuffers = nullptr;
            submit_info.signalSemaphoreCount = 0;
            submit_info.pSignalSemaphores = nullptr;
        }

        vk_graphics_queue.submit({submit_info}, *vk_in_flight_fence);
        return true;
    }

    [[nodiscard]] bool Present(std::uint32_t image_index) override {
        LOGIFACE_LOG(trace, "entering Present with image index " + std::to_string(image_index));
        if (!device_ || !swapchain_) return false;

        const vk::raii::Queue& vk_graphics_queue = device_->GetGraphicsQueue();
        const vk::raii::Semaphore& vk_render_finished_semaphore = GetRenderFinishedSemaphore(image_index);

        vk::PresentInfoKHR present_info{};
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &*vk_render_finished_semaphore;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &*swapchain_->GetSwapchain();
        present_info.pImageIndices = &image_index;

        try {
            const vk::Result present = vk_graphics_queue.presentKHR(present_info);
            if (present == vk::Result::eErrorOutOfDateKHR || present == vk::Result::eSuboptimalKHR) {
                return false;
            }
            if (present != vk::Result::eSuccess) {
                return false;
            }
        } catch (...) {
            return false;
        }

        swapchain_->GetImageInitializedFlags()[image_index] = true;
        LOGIFACE_LOG(trace, "leaving Present successfully.");
        return true;
    }

    // Corrected lifetime primitive: frame N is complete when the frame has
    // actually been submitted into slot N % FIF (submission record) AND that
    // slot's fence is signaled. The old residue-only query was false forever at
    // FIF == 1 because SubmitFrame resets the just-waited fence, and true for a
    // recorded-but-dropped frame whose runs were never submitted.
    [[nodiscard]] bool IsFrameComplete(std::uint32_t frame_idx) override {
        if (!device_ || !swapchain_) return false;
        // The record for this slot must name this frame (or a later frame that
        // reused the slot, which proves this frame's submit happened); a
        // stale/never-submitted slot fails first.
        if (!submission_record_.IsRecorded(frame_idx)) {
            return false;
        }
        try {
            const auto status = device_->GetDevice().waitForFences(
                *device_->GetInFlightFence(frame_idx), vk::True, 0);
            return status == vk::Result::eSuccess;
        } catch (...) {
            return false;
        }
    }

    void WaitDeviceIdle() override {
        if (!device_ || !device_->IsValid()) return;
        // Serialize with every other device-scope waiter: vkDeviceWaitIdle
        // touches all queues, which the API requires to be externally
        // synchronized. See VulkanShared.DeviceScope.
        VulkanShared::DeviceScopeGuard guard;
        device_->GetDevice().waitIdle();
    }

    void Shutdown() override {
        {
            auto s = DebugSection("wait for device idle before destroying resources in VulkanBootstrapBackend::Shutdown");
            WaitDeviceIdle();
        }

        {
            auto s = DebugSection("destroy render finished semaphores in VulkanBootstrapBackend::Shutdown");
            render_finished_semaphores_.clear();
        }
        if (swapchain_) {
            auto s = DebugSection("shutdown swapchain in VulkanBootstrapBackend::Shutdown");
            swapchain_->Shutdown();
        }
        if (device_) {
            auto s = DebugSection("shutdown device in VulkanBootstrapBackend::Shutdown");
            device_->Shutdown();
        }
        if (instance_) {
            auto s = DebugSection("shutdown instance in VulkanBootstrapBackend::Shutdown");
            instance_->Shutdown();
        }
    }

private:
    std::unique_ptr<VulkanInstance> instance_{};
    std::unique_ptr<VulkanDevice> device_{};

    std::unique_ptr<VulkanSwapchain> swapchain_{};
    std::vector<std::unique_ptr<vk::raii::Semaphore>> render_finished_semaphores_{};
    std::vector<IVulkanBootstrap::QueueRunSubmit> frame_runs_{};
    // Per-FIF submission record, see FrameSubmissionRecord.cppm.
    FrameSubmissionRecord submission_record_{};
    std::uint32_t current_image_index_ = 0;
    VulkanBootstrapConfig config_{};
    std::string error_message_{};
    bool requirements_unmet_ = false;
};

}  // namespace

std::shared_ptr<IVulkanBootstrap> CreateVulkanBootstrapBackend() {
    return std::make_shared<DefaultVulkanBootstrapBackend>();
}

}  // namespace VulkanBackend::Vulkan
