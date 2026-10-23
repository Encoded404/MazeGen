module;

export module MazeGenerator.MazeRenderPass;

import std;
import vulkan;

import VulkanEngine.PipelinePass;

export namespace MazeGenerator {

// 48-byte fragment push constant. The three arrays keep the same 16-byte
// alignment the Slang float4/uint4 fields use.
struct alignas(16) MazeConstants {
    std::array<float, 4> view{};   // center_x, center_y, pixels_per_cell, wall_thickness
    std::array<float, 4> screen{}; // width, height, tier (0/1/2), density block size
    std::array<std::uint32_t, 4> maze{}; // width, height, tiles_x, density mip count
};

static_assert(sizeof(MazeConstants) == 48);

// Fullscreen app pass that renders the maze bits (and the coarse density
// pyramid) directly into the backbuffer.
class MazeRenderPass final : public VulkanEngine::PipelinePass::IPipelinePass {
public:
    MazeRenderPass(std::uint64_t vertex_shader, std::uint64_t fragment_shader);
    ~MazeRenderPass() override;

    MazeRenderPass(const MazeRenderPass&) = delete;
    MazeRenderPass& operator=(const MazeRenderPass&) = delete;

    [[nodiscard]] std::string_view GetName() const override { return "maze-render"; }

    void Setup(VulkanEngine::PipelinePass::PassSetupContext& ctx) override;
    void Execute(const VulkanEngine::PipelinePass::FrameContext& ctx,
                 vk::CommandBuffer cmd) override;

    void SetConstants(const MazeConstants& constants) noexcept { constants_ = constants; }

private:
    std::uint64_t vertex_shader_ = 0;
    std::uint64_t fragment_shader_ = 0;
    MazeConstants constants_{};
};

} // namespace MazeGenerator
