// Per-pass GPU timestamps for the single-graphics-queue frame.
//
// Two timestamp queries bracket every render-graph pass that has an execute
// callback, and the pair for a (ring slot, pass) lives in a dedicated slot of
// one pool. A ring's results are therefore only read once that ring's prior
// submission is fence-complete, which is the same guarantee the pipeline
// statistics pool relies on: AcquireNextImage has already waited on the fence
// `frames_in_flight` back.
//
// It is deliberately single-queue. Timestamps cannot be written on a queue
// whose family reports `timestampValidBits == 0`, and a query pool cannot span
// queues, so the async-compute path leaves this off rather than measuring half
// a frame.

module;

#include <logging/logging_macros.hpp>

export module VulkanBackend.Vulkan.GpuPassTimer;

import std;

import logiface;

import vulkan;

export namespace VulkanBackend::Vulkan {

// One pass's GPU time, as measured by the timestamp pair that bracketed it.
struct GpuPassSample {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::string name{};
    double ms = 0.0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

class GpuPassTimer {
public:
    // Passes past this index are simply not timed. The engine's compiled graphs
    // are an order of magnitude below it, and the pool is small either way.
    static constexpr std::uint32_t kMaxPasses = 256;
    // Reserved index for the work the Renderer records ahead of the graph
    // (staged uploads, descriptor writes, the physical-camera composite). It has
    // no CompiledPass, so it would otherwise be invisible to the frame's total.
    static constexpr std::uint32_t kPrepPassIndex = kMaxPasses - 1;

    GpuPassTimer() = default;
    ~GpuPassTimer() = default;

    GpuPassTimer(const GpuPassTimer&) = delete;
    GpuPassTimer& operator=(const GpuPassTimer&) = delete;

    // Creates the timestamp pool. Returns false (and stays disabled) when the
    // graphics queue family cannot write timestamps.
    bool Initialize(const vk::raii::PhysicalDevice& physical_device,
                    const vk::raii::Device& device,
                    std::uint32_t graphics_queue_family,
                    std::uint32_t frames_in_flight) {
        if (pool_) {
            return true;
        }
        frames_in_flight_ = std::max<std::uint32_t>(frames_in_flight, 1u);

        const auto families = physical_device.getQueueFamilyProperties();
        if (graphics_queue_family >= families.size() ||
            families[graphics_queue_family].timestampValidBits == 0) {
            LOGIFACE_LOG(warn,
                         "GpuPassTimer: the graphics queue family reports no timestamp "
                         "support; per-pass GPU timing is off");
            return false;
        }

        const float period =
            static_cast<float>(physical_device.getProperties().limits.timestampPeriod);
        period_ns_ = period > 0.0f ? period : 1.0f;

        const std::uint32_t query_count = frames_in_flight_ * kMaxPasses * 2u;
        vk::QueryPoolCreateInfo info{};
        info.queryType = vk::QueryType::eTimestamp;
        info.queryCount = query_count;
        pool_ = std::make_unique<vk::raii::QueryPool>(device, info);
        device_ = &device;

        const vk::Device raw_device = *device;
        raw_device.resetQueryPool(**pool_, 0, query_count);

        slot_names_.assign(static_cast<std::size_t>(frames_in_flight_) * kMaxPasses,
                           std::string{});
        slot_pass_count_.assign(frames_in_flight_, 0u);
        slot_written_.assign(static_cast<std::size_t>(frames_in_flight_) * kMaxPasses, 0u);

        LOGIFACE_LOG(info,
                     std::format("GpuPassTimer: per-pass GPU timestamps enabled ({} slots, "
                                 "{:.3f} ns/tick)",
                                 query_count, period_ns_));
        return true;
    }

    [[nodiscard]] bool Enabled() const { return pool_ != nullptr; }

    // Selects the frames-in-flight slot this frame records into. Collect() reads
    // that slot's previous content; the writes that follow overwrite it.
    void BeginFrame(std::uint32_t ring_index) {
        ring_index_ = frames_in_flight_ == 0 ? 0 : ring_index % frames_in_flight_;
    }

    void WriteBegin(vk::CommandBuffer command_buffer, std::uint32_t pass_index,
                    std::string_view name) {
        if (!pool_ || pass_index >= kMaxPasses ||
            ring_index_ >= slot_pass_count_.size()) {
            return;
        }
        command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands,
                                       **pool_, QueryIndex(ring_index_, pass_index));
        const auto slot =
            static_cast<std::size_t>(ring_index_) * kMaxPasses + pass_index;
        if (slot < slot_names_.size()) {
            slot_names_[slot].assign(name);
            slot_written_[slot] = 1u;
        }
        slot_pass_count_[ring_index_] =
            std::max(slot_pass_count_[ring_index_], pass_index + 1u);
    }

    // The Renderer's pre-graph work (see kPrepPassIndex).
    void WritePrepBegin(vk::CommandBuffer command_buffer) {
        WriteBegin(command_buffer, kPrepPassIndex, "frame-prep (uploads + composite)");
    }
    void WritePrepEnd(vk::CommandBuffer command_buffer) {
        WriteEnd(command_buffer, kPrepPassIndex);
    }

    void WriteEnd(vk::CommandBuffer command_buffer, std::uint32_t pass_index) {
        if (!pool_ || pass_index >= kMaxPasses ||
            ring_index_ >= slot_pass_count_.size()) {
            return;
        }
        command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands,
                                       **pool_, QueryIndex(ring_index_, pass_index) + 1u);
    }

    // Reads the selected slot, labelling each sample with the pass name recorded
    // when it was written, then clears the slot for reuse. Call only once that
    // slot's submission is complete.
    void Collect(std::vector<GpuPassSample>& out) {
        out.clear();
        if (!pool_ || ring_index_ >= slot_pass_count_.size()) {
            return;
        }
        const std::uint32_t ring = ring_index_;
        const std::uint32_t passes = slot_pass_count_[ring];
        if (passes == 0) {
            return;
        }

        // One read per written pass. A single ranged read would return
        // VK_NOT_READY as soon as it covered a query that was never written
        // (every index between the graph's passes and the reserved prep index),
        // which would discard the results that are available.
        const vk::raii::Device& device = *device_;
        for (std::uint32_t pass = 0; pass < passes; ++pass) {
            const auto slot = static_cast<std::size_t>(ring) * kMaxPasses + pass;
            if (slot_written_[slot] == 0u) {
                continue; // reserved/unused index, never written
            }
            std::array<std::uint64_t, 2> words{};
            const vk::Result result =
                static_cast<vk::Result>(device.getDispatcher()->vkGetQueryPoolResults(
                    static_cast<vk::Device::CType>(*device),
                    static_cast<vk::QueryPool::CType>(**pool_),
                    QueryIndex(ring, pass), 2u,
                    static_cast<vk::DeviceSize>(2u * sizeof(std::uint64_t)), words.data(),
                    static_cast<vk::DeviceSize>(sizeof(std::uint64_t)),
                    static_cast<vk::QueryResultFlags::MaskType>(
                        vk::QueryResultFlagBits::e64)));
            if (result != vk::Result::eSuccess) {
                continue; // not complete yet
            }
            const std::uint64_t begin = words[0];
            const std::uint64_t end = words[1];
            if (end < begin) {
                continue; // an end that never landed
            }
            GpuPassSample sample{};
            sample.name = slot_names_[slot];
            sample.ms = static_cast<double>(end - begin) *
                        static_cast<double>(period_ns_) * 1.0e-6;
            out.push_back(std::move(sample));
        }
        // The slot's prior submission is complete, so its queries can be reused.
        const vk::Device raw_device = *device;
        raw_device.resetQueryPool(**pool_, QueryIndex(ring, 0), passes * 2u);
        for (std::uint32_t pass = 0; pass < passes; ++pass) {
            slot_written_[static_cast<std::size_t>(ring) * kMaxPasses + pass] = 0u;
        }
        slot_pass_count_[ring] = 0;
    }

    void Shutdown() {
        pool_.reset();
        device_ = nullptr;
        slot_names_.clear();
        slot_pass_count_.clear();
        slot_written_.clear();
    }

private:
    [[nodiscard]] std::uint32_t QueryIndex(std::uint32_t ring,
                                           std::uint32_t pass) const {
        return (ring * kMaxPasses + pass) * 2u;
    }

    std::unique_ptr<vk::raii::QueryPool> pool_{};
    // Kept only for the raw dispatcher used by the readback, matching GpuStats.
    const vk::raii::Device* device_ = nullptr;
    float period_ns_ = 1.0f;
    std::uint32_t frames_in_flight_ = 1;
    std::uint32_t ring_index_ = 0;
    // Pass names recorded into each (slot, pass), so Collect can label its samples.
    std::vector<std::string> slot_names_{};
    // 1 for a (slot, pass) that had a begin written this use; Collect reads only
    // those, so a reserved or idle index never turns into a phantom 0 ms sample.
    std::vector<std::uint8_t> slot_written_{};
    std::vector<std::uint32_t> slot_pass_count_{};
};

} // namespace VulkanBackend::Vulkan
