module;

#include <cassert>

export module VulkanEngine.PipelineCompiler;

import std;
import vulkan;

import VulkanEngine.PipelineFactory;
import VulkanEngine.ShaderManager;

export namespace VulkanEngine::ShaderSystem {

using PipelineOutcome = std::expected<PipelineProduct, PipelineError>;

// ── Executor seam ────────────────────────────────────────────────────────────
//
// The compiler partitions a submission window by desc kind and hands each
// partition to an executor that owns the actual Vulkan calls. The factory
// executor routes into PipelineFactory's batched entry points (one driver
// call per partition). Tests inject a fake executor to verify window
// semantics without a device; the async phase will reuse the seam.
class IPipelineBatchExecutor {
public:
    virtual ~IPipelineBatchExecutor() = default;

    // One outcome per desc, in desc order. An executor may not return
    // fewer/more; a size mismatch is treated as executor failure by the
    // compiler (every request of the group gets an error outcome).
    [[nodiscard]] virtual std::vector<PipelineOutcome>
        ExecuteCompute(ShaderManager& shaders,
                       std::span<const ComputePipelineDesc> descs) = 0;
    [[nodiscard]] virtual std::vector<PipelineOutcome>
        ExecuteGraphics(ShaderManager& shaders,
                        std::span<const GraphicsPipelineDesc> descs) = 0;
};

class PipelineFactoryExecutor final : public IPipelineBatchExecutor {
public:
    explicit PipelineFactoryExecutor(PipelineFactory& factory) : factory_(factory) {}

    [[nodiscard]] std::vector<PipelineOutcome>
        ExecuteCompute(ShaderManager& shaders,
                       std::span<const ComputePipelineDesc> descs) override {
        return factory_.CreateComputeBatch(descs, shaders);
    }

    [[nodiscard]] std::vector<PipelineOutcome>
        ExecuteGraphics(ShaderManager& shaders,
                        std::span<const GraphicsPipelineDesc> descs) override {
        return factory_.CreateGraphicsBatch(descs, shaders);
    }

private:
    PipelineFactory& factory_;
};

// ── Submission window ────────────────────────────────────────────────────────
//
// Producers submit declarative pipeline requests (desc + delivery callback)
// during initialization or per-frame compilation, then Flush once: the whole
// window compiles in batched driver calls instead of one call per pipeline.
//
// THREADING CONTRACT (main thread only, phases 1–3):
//   - Submit() and Flush() run on the main thread.
//   - Delivery callbacks run on the thread that calls Flush (the main
//     thread), exactly once per submitted request, in submission order
//     grouped by partition (all compute, then all graphics).
//   - The async phase will keep deliveries on the main thread via a frame
//     pump; PipelineSlot::Swap stays main-thread-only by design.
class PipelineCompiler {
public:
    struct Request {
        std::variant<ComputePipelineDesc, GraphicsPipelineDesc> desc;
        // Log-friendly identity, carried into delivery errors when the
        // executor cannot attribute a failure (defensive path only).
        std::string_view debug_name;
        std::function<void(PipelineOutcome&&)> deliver;
    };

    explicit PipelineCompiler(IPipelineBatchExecutor& executor) : executor_(executor) {}

    void Submit(Request request) {
        assert(request.deliver); // exactly-once delivery requires a callback
        requests_.push_back(std::move(request));
    }

    void SubmitCompute(ComputePipelineDesc desc, std::string_view debug_name,
                       std::function<void(PipelineOutcome&&)> deliver) {
        Submit(Request{std::move(desc), debug_name, std::move(deliver)});
    }

    void SubmitGraphics(GraphicsPipelineDesc desc, std::string_view debug_name,
                        std::function<void(PipelineOutcome&&)> deliver) {
        Submit(Request{std::move(desc), debug_name, std::move(deliver)});
    }

    // Compile everything submitted since the last Flush. Partitions by desc
    // kind, runs one executor call per kind (a batched driver call), then
    // delivers per-request outcomes and empties the window. An empty window
    // is a no-op that calls no executor method.
    void Flush(ShaderManager& shaders) {
        if (requests_.empty()) return;

        try {
            std::vector<ComputePipelineDesc> compute_descs;
            std::vector<std::size_t> compute_indices;
            std::vector<GraphicsPipelineDesc> graphics_descs;
            std::vector<std::size_t> graphics_indices;
            for (std::size_t i = 0; i < requests_.size(); ++i) {
                std::visit([&]<typename T>(T& d) {
                    if constexpr (std::same_as<T, ComputePipelineDesc>) {
                        compute_descs.push_back(d);
                        compute_indices.push_back(i);
                    } else {
                        graphics_descs.push_back(d);
                        graphics_indices.push_back(i);
                    }
                }, requests_[i].desc);
            }

            if (!compute_descs.empty()) {
                auto outcomes = executor_.ExecuteCompute(shaders, compute_descs);
                DeliverGroup(outcomes, compute_indices);
            }
            if (!graphics_descs.empty()) {
                auto outcomes = executor_.ExecuteGraphics(shaders, graphics_descs);
                DeliverGroup(outcomes, graphics_indices);
            }
        } catch (const std::exception& e) {
            // Executor threw outside its contract: fail the whole window
            // rather than leaving requests silently undelivered.
            PipelineError error{
                PipelineErrorKind::DriverRejectedPipeline,
                vk::Result::eErrorInitializationFailed,
                std::format("pipeline batch executor threw: {}", e.what())};
            for (auto& request : requests_) {
                if (request.deliver) {
                    // The defensive error mentions the request so a producer
                    // log line can identify which window entry died.
                    error.message += std::format(" [{}]", request.debug_name);
                    request.deliver(std::unexpected(error));
                }
            }
        }
        requests_.clear();
    }

    [[nodiscard]] std::size_t Pending() const noexcept { return requests_.size(); }

    // Drop unflushed submissions without compiling or delivering (teardown).
    void Clear() noexcept { requests_.clear(); }

private:
    // Enforce the executor contract, then deliver each outcome to its request.
    void DeliverGroup(std::vector<PipelineOutcome>& outcomes,
                      const std::vector<std::size_t>& indices) {
        if (outcomes.size() != indices.size()) {
            PipelineError error{
                PipelineErrorKind::DriverRejectedPipeline,
                vk::Result::eErrorInitializationFailed,
                std::format("pipeline batch executor returned {} outcomes for {} descs",
                            outcomes.size(), indices.size())};
            for (const std::size_t i : indices) {
                if (requests_[i].deliver) {
                    requests_[i].deliver(std::unexpected(error));
                }
            }
            return;
        }
        for (std::size_t j = 0; j < indices.size(); ++j) {
            if (requests_[indices[j]].deliver) {
                requests_[indices[j]].deliver(std::move(outcomes[j]));
            }
        }
    }

    IPipelineBatchExecutor& executor_;
    std::vector<Request> requests_;
};

} // namespace VulkanEngine::ShaderSystem