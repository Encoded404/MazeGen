// Alpha-blended fullscreen overlay that draws the solver's debug state on top
// of the maze pass. It is a separate pass so it can be enabled/disabled
// independently, hot-reload independently, and draw at any maze tier.

module;

export module MazeGenerator.DebugOverlayPass;

import std;
import vulkan;

import VulkanEngine.PipelinePass;

export namespace MazeGenerator {

// 64-byte fragment push constant. The arrays keep the 16-byte alignment the
// Slang float4/uint4 fields use.
struct alignas(16) DebugConstants {
    std::array<float, 4> view{};   // center_x, center_y, pixels_per_cell, alpha
    std::array<float, 4> screen{}; // width, height, tier (0/1/2), block size
    std::array<std::uint32_t, 4> grid{};  // width, height, blocks_x, category mask
    std::array<std::uint32_t, 4> extra{}; // start cell, goal cell, flags, reserved
};

static_assert(sizeof(DebugConstants) == 64);

class DebugOverlayPass final : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    DebugOverlayPass(std::uint64_t vertex_shader, std::uint64_t fragment_shader);
    ~DebugOverlayPass() override;

    DebugOverlayPass(const DebugOverlayPass&) = delete;
    DebugOverlayPass& operator=(const DebugOverlayPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override {
        return "maze-debug-overlay";
    }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

    void SetConstants(const DebugConstants& constants) noexcept {
        constants_ = constants;
    }

private:
    std::uint64_t vertex_shader_ = 0;
    std::uint64_t fragment_shader_ = 0;
    DebugConstants constants_{};
};

} // namespace MazeGenerator
