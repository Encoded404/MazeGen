module;

#include <SDL3/SDL_video.h>

export module VulkanBackend.Platform.SdlPlatform;

import std;

import VulkanBackend.Event;
import VulkanShared.CallbackList;

export namespace VulkanBackend::Platform {

enum class PlatformStatus : std::uint8_t {
    Ok,
    NotInitialized,
    BackendInitFailed,
    WindowCreateFailed,
    QuitRequested,
    FatalError
};

struct PlatformConfig {
    std::string window_title = "MazeGenerator"; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t window_width = 1280; // NOLINT(misc-non-private-member-variables-in-classes)
    std::uint32_t window_height = 720; // NOLINT(misc-non-private-member-variables-in-classes)
};

struct PlatformState {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    bool initialized = false;
    bool window_created = false;
    bool quit_requested = false;
    bool minimized = false;
    bool resized = false;
    std::uint32_t drawable_width = 0;
    std::uint32_t drawable_height = 0;
    // Physical pixels per logical point for the window's current display. The
    // window size, every input coordinate and every UI size a caller writes are
    // in logical points; the swapchain extent is in physical pixels, and this is
    // the ratio between them. It is 1.0 wherever the platform reports no scale,
    // which is also what a backend without a window must report.
    float content_scale = 1.0f;
    PlatformStatus status = PlatformStatus::NotInitialized;
    std::string error_message; // Optional detailed error message for fatal errors
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

class IPlatformBackend {
public:
    virtual ~IPlatformBackend() = default;

    [[nodiscard]] virtual bool Initialize() = 0;
    virtual void Shutdown() = 0;
    [[nodiscard]] virtual bool CreateMainWindow(const PlatformConfig& config) = 0;
    [[nodiscard]] virtual VulkanBackend::Event::EventList PumpEvents() = 0;
    [[nodiscard]] virtual SDL_Window* GetNativeWindowHandle() const = 0;

    // Physical pixels per logical point on the window's current display.
    // Defaulted rather than pure so a backend with no display to ask -- a test
    // double, a headless shell -- reports 1.0 instead of having to answer a
    // question it cannot.
    [[nodiscard]] virtual float GetDisplayScale() const { return 1.0f; }

    virtual VulkanShared::CallbackList<void(void*)>& GetSdlEventProcessors() = 0;
};

class SdlPlatform {
public:
    explicit SdlPlatform(std::shared_ptr<IPlatformBackend> backend);

    [[nodiscard]] bool Initialize(const PlatformConfig& config);
    void Shutdown();
    [[nodiscard]] VulkanBackend::Event::EventList PollEvents();

    [[nodiscard]] const PlatformState& GetState() const;
    [[nodiscard]] bool IsInitialized() const;
    [[nodiscard]] bool ShouldQuit() const;
    [[nodiscard]] SDL_Window* GetNativeWindowHandle() const;
    [[nodiscard]] IPlatformBackend& GetBackend() const;

private:
    std::shared_ptr<IPlatformBackend> backend_{};
    PlatformState state_{};
};

}  // namespace VulkanBackend::Platform

