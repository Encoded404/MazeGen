module;

#include <logging/logging_macros.hpp>

module VulkanBackend.Vulkan.VulkanBootstrap;

import std;
import std.compat;

import logiface;

import VulkanShared.ScopedSection;

namespace VulkanBackend::Vulkan {

namespace {

// Debug-level timing for the driver-level bootstrap steps. Sections live at
// this orchestration level (not inside the backend methods) because
// CreateSwapchain is also driven by RecreateSwapchain at runtime — a startup
// section must not log on mid-run swapchain recreation.
VulkanShared::ScopedSection DebugSection(const std::string& name) {
    return VulkanShared::ScopedSection{name, [](const std::string& section, double ms) {
        LOGIFACE_LOG(debug, section + ": " + std::to_string(ms) + " ms");
    }};
}

} // anonymous namespace

VulkanBootstrap::VulkanBootstrap(std::shared_ptr<IVulkanBootstrap> backend)
    : backend_(std::move(backend)) {}

bool VulkanBootstrap::Initialize(const VulkanBootstrapConfig& config) {
    if (!backend_) {
        snapshot_.status = BootstrapStatus::FatalError;
        initialized_ = false;
        return false;
    }

    config_ = config;
    config_.frames_in_flight = std::max(config_.frames_in_flight, 1u);
    config_.preferred_swapchain_image_count = std::max(config_.preferred_swapchain_image_count, 2u);

    snapshot_ = VulkanBootstrapState{};
    snapshot_.frames_in_flight = config_.frames_in_flight;

    // Each step's scope is the if-statement: on success the section closes right
    // after the step; on failure it also covers the in-branch partial shutdown
    // before the false return.
    if (auto s = DebugSection("engineinit.vulkan_instance"); !backend_->CreateInstance(config_)) {
        snapshot_.status = backend_->HasUnmetRequirements()
            ? BootstrapStatus::RequirementsUnmet
            : BootstrapStatus::InstanceCreationFailed;
        snapshot_.error_message = backend_->GetErrorMessage();
        backend_->Shutdown();
        return false;
    }
    snapshot_.instance_ready = true;

    if (auto s = DebugSection("engineinit.physical_device"); !backend_->SelectPhysicalDevice()) {
        snapshot_.status = backend_->HasUnmetRequirements()
            ? BootstrapStatus::RequirementsUnmet
            : BootstrapStatus::DeviceSelectionFailed;
        snapshot_.error_message = backend_->GetErrorMessage();
        backend_->Shutdown();
        return false;
    }

    if (auto s = DebugSection("engineinit.vulkan_device"); !backend_->CreateLogicalDevice(config_.frames_in_flight)) {
        snapshot_.status = backend_->HasUnmetRequirements()
            ? BootstrapStatus::RequirementsUnmet
            : BootstrapStatus::DeviceCreationFailed;
        snapshot_.error_message = backend_->GetErrorMessage();
        backend_->Shutdown();
        return false;
    }
    snapshot_.device_ready = true;

    std::uint32_t swapchain_image_count = 0;
    if (auto s = DebugSection("engineinit.swapchain");
        !backend_->CreateSwapchain(config_.preferred_swapchain_image_count,
                                   config_.present_mode, swapchain_image_count)) {
        snapshot_.status = BootstrapStatus::SwapchainCreationFailed;
        backend_->Shutdown();
        return false;
    }

    snapshot_.swapchain_ready = true;
    snapshot_.swapchain_image_count = swapchain_image_count;
    if (!backend_->GetSwapchainExtent(snapshot_.swapchain_width, snapshot_.swapchain_height)) {
        snapshot_.status = BootstrapStatus::SwapchainCreationFailed;
        backend_->Shutdown();
        return false;
    }
    snapshot_.status = BootstrapStatus::Ok;

    pending_status_ = BootstrapStatus::Ok;
    initialized_ = true;

    return true;
}

void VulkanBootstrap::Shutdown() {
    if (backend_ && initialized_) {
        backend_->Shutdown();
    }

    snapshot_ = VulkanBootstrapState{};
    snapshot_.status = BootstrapStatus::NotInitialized;
    pending_status_ = BootstrapStatus::NotInitialized;
    initialized_ = false;
}

VulkanBootstrapState VulkanBootstrap::BeginFrame() {
    LOGIFACE_LOG(trace, "entering VulkanBootstrap::BeginFrame");
    if (!initialized_) {
        VulkanBootstrapState snapshot{};
        snapshot.status = BootstrapStatus::NotInitialized;
        return snapshot;
    }

    if (pending_status_ == BootstrapStatus::DeviceLost) {
        snapshot_.status = BootstrapStatus::DeviceLost;
        return snapshot_;
    }

    if (pending_status_ == BootstrapStatus::SwapchainOutOfDate) {
        snapshot_.status = BootstrapStatus::SwapchainOutOfDate;
        return snapshot_;
    }

    snapshot_.status = BootstrapStatus::Ok;
    LOGIFACE_LOG(trace, "leaving VulkanBootstrap::BeginFrame successfully");
    return snapshot_;
}

void VulkanBootstrap::EndFrame() {
    LOGIFACE_LOG(trace, "entering VulkanBootstrap::EndFrame");
    if (!initialized_) {
        return;
    }

    ++snapshot_.frame_index;
    LOGIFACE_LOG(trace, "leaving VulkanBootstrap::EndFrame. frame_index is " + std::to_string(snapshot_.frame_index));
}

void VulkanBootstrap::NotifySwapchainOutOfDate() {
    pending_status_ = BootstrapStatus::SwapchainOutOfDate;
}

bool VulkanBootstrap::RecreateSwapchain() {
    if (!initialized_ || !backend_) {
        return false;
    }

    std::uint32_t swapchain_image_count = 0;
    if (!backend_->CreateSwapchain(config_.preferred_swapchain_image_count, config_.present_mode, swapchain_image_count)) {
        snapshot_.status = BootstrapStatus::SwapchainCreationFailed;
        snapshot_.swapchain_ready = false;
        return false;
    }

    snapshot_.swapchain_ready = true;
    snapshot_.swapchain_image_count = swapchain_image_count;
    if (!backend_->GetSwapchainExtent(snapshot_.swapchain_width, snapshot_.swapchain_height)) {
        snapshot_.status = BootstrapStatus::SwapchainCreationFailed;
        snapshot_.swapchain_ready = false;
        return false;
    }
    snapshot_.status = BootstrapStatus::Ok;
    pending_status_ = BootstrapStatus::Ok;
    return true;
}

bool VulkanBootstrap::AcquireNextImage(std::uint32_t& out_image_index) {
    if (!initialized_ || !backend_) {
        return false;
    }
    return backend_->AcquireNextImage(snapshot_.frame_index, out_image_index);
}

bool VulkanBootstrap::SubmitFrame(const std::uint32_t image_index, const bool rendering_succeeded) {
    if (!initialized_ || !backend_) {
        return false;
    }
    // The frame counter in the snapshot is the monotonic iteration index; the
    // submit targets the per-slot resources indexed by `frame_index % FIF`.
    return backend_->SubmitFrame(snapshot_.frame_index, image_index, rendering_succeeded);
}

bool VulkanBootstrap::Present(const std::uint32_t image_index) {
    if (!initialized_ || !backend_) {
        return false;
    }
    return backend_->Present(image_index);
}

bool VulkanBootstrap::IsFrameComplete(const std::uint32_t frame_idx) const {
    if (!initialized_ || !backend_) {
        return false;
    }
    return backend_->IsFrameComplete(frame_idx);
}

void VulkanBootstrap::NotifyDeviceLost() {
    pending_status_ = BootstrapStatus::DeviceLost;
}

bool VulkanBootstrap::IsInitialized() const {
    return initialized_;
}

const VulkanBootstrapState& VulkanBootstrap::GetSnapshot() const {
    return snapshot_;
}

}  // namespace VulkanBackend::Vulkan
