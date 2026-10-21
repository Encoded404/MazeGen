// Per-ECS-system CPU timing interface.
//
// engine/core/ecs must be able to report system timings without importing the
// runtime (layering is core -> shared, runtime -> core), so the receiver lives
// here as a small dependency-free interface. Runtime::FrameProfiler implements
// it; a null sink means the schedules read no clock at all.

export module VulkanShared.SystemProfile;

import std;

export namespace VulkanShared {

enum class SystemProfileKind : std::uint8_t {
    Frame = 0,      // a FrameSchedule system (runs on the rendered-frame path)
    Component = 1,  // a SystemSchedule system (runs on the fixed-step path)
};

[[nodiscard]] constexpr std::string_view SystemProfileKindName(
    const SystemProfileKind kind) noexcept {
    switch (kind) {
        case SystemProfileKind::Frame: return "frame";
        case SystemProfileKind::Component: return "component";
    }
    return "unknown";
}

// Non-owning per-system timing receiver. `label` is valid only for the call;
// an implementation that keeps it must copy it.
class ISystemProfileSink {
public:
    virtual ~ISystemProfileSink() = default;
    virtual void AddSystemTiming(SystemProfileKind kind, std::string_view label,
                                 double milliseconds) = 0;
};

} // namespace VulkanShared
