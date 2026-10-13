module;

export module VulkanEngine.GpuResources.TransientAllocator;

import std;
import std.compat;

import vulkan;

export import VulkanBackend.Vulkan.VulkanBootstrap;

export namespace VulkanEngine::GpuResources {

enum class TransientKind : std::uint8_t {
    Image,
    Buffer
};

// A transient resource's allocation requirements and intra-frame lifetime.
// `heap_key` groups resources that may share one heap (e.g. kind + memory type);
// resources only ever alias within the same heap.
struct TransientRequirements {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::string name{};
    TransientKind kind = TransientKind::Buffer;
    std::uint32_t heap_key = 0;
    std::uint64_t size = 0;
    std::uint64_t alignment = 1;
    std::int32_t first_pass = 0;
    std::int32_t last_pass = 0;
    // Aliasable transients are created with VK_IMAGE_CREATE_ALIAS_BIT and carry
    // an Undefined initial layout. Only a pair of aliasable resources may share
    // one address range.
    bool aliasable = false;
    // Inactive entries are index placeholders so that a desc's position matches
    // its graph resource index; they are never allocated or placed.
    bool active = true;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct TransientPlacement {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t resource_index = 0;
    std::uint32_t fif_slot = 0;
    std::uint32_t heap_key = 0;
    std::uint64_t offset = 0;   // within one FIF copy of the heap
    std::uint64_t size = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// B reuses the memory A occupied. The executor must order B's first use after
// A's last use; a discard barrier alone is not sufficient.
struct PlannedAlias {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::uint32_t aliased_resource = 0;
    std::uint32_t after_resource = 0;
    std::uint32_t heap_key = 0;
    std::int32_t pass_index = 0;
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

struct TransientPlacementPlan {
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    bool valid = false;
    std::uint32_t frames_in_flight = 1;
    std::vector<TransientPlacement> placements{};
    std::vector<PlannedAlias> aliases{};
    // heap_key -> bytes for one FIF copy (multiply by frames_in_flight for the
    // total backing size).
    std::vector<std::pair<std::uint32_t, std::uint64_t>> heap_copy_sizes{};
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] std::uint64_t GetHeapCopySize(std::uint32_t heap_key) const {
        for (const auto& [key, size] : heap_copy_sizes) {
            if (key == heap_key) {
                return size;
            }
        }
        return 0;
    }
};

namespace detail {

[[nodiscard]] inline std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    if (alignment <= 1) {
        return value;
    }
    const std::uint64_t mod = value % alignment;
    return mod == 0 ? value : value + (alignment - mod);
}

// One address range handed out at some point in the frame. `free_after_pass` is
// the pass after which the occupant no longer touches it, so the range may be
// reused by a resource whose first use is later.
struct ReusableRange {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::int32_t free_after_pass = -1;
    std::uint32_t origin_resource = std::numeric_limits<std::uint32_t>::max();
    bool origin_aliasable = false;
};

}  // namespace detail

// Pure interval-based placement. For each heap it lays resources out once,
// reusing an address range only between aliasable resources with disjoint
// [first_pass, last_pass] lifetimes, then replicates the layout into
// `frames_in_flight` copies. No device access, so this is unit-testable
// everywhere; the device heaps execute the resulting plan.
[[nodiscard]] inline TransientPlacementPlan PlanTransientPlacements(
    const std::vector<TransientRequirements>& requirements,
    std::uint32_t frames_in_flight) {
    using detail::AlignUp;
    using detail::ReusableRange;

    TransientPlacementPlan plan{};
    plan.frames_in_flight = std::max(1u, frames_in_flight);
    plan.valid = true;

    std::vector<std::uint32_t> heap_keys{};
    for (const auto& requirement : requirements) {
        if (!requirement.active) {
            continue;
        }
        if (std::ranges::find(heap_keys, requirement.heap_key) == heap_keys.end()) {
            heap_keys.push_back(requirement.heap_key);
        }
    }

    for (const std::uint32_t heap_key : heap_keys) {
        std::vector<std::uint32_t> group{};
        for (std::uint32_t index = 0; index < requirements.size(); ++index) {
            if (requirements[index].active && requirements[index].heap_key == heap_key) {
                group.push_back(index);
            }
        }

        std::ranges::sort(group, [&](std::uint32_t a, std::uint32_t b) {
            if (requirements[a].first_pass != requirements[b].first_pass) {
                return requirements[a].first_pass < requirements[b].first_pass;
            }
            return a < b;
        });

        std::vector<ReusableRange> ranges{};
        std::vector<std::pair<std::uint32_t, std::uint64_t>> offsets{};
        std::uint64_t copy_end = 0;

        for (const std::uint32_t index : group) {
            const auto& requirement = requirements[index];
            const std::uint64_t size = std::max<std::uint64_t>(requirement.size, 1);

            // A range is reusable once its occupant finished strictly before
            // this resource's first pass.
            const auto reusable = [&](const ReusableRange& range) {
                if (range.free_after_pass >= requirement.first_pass) {
                    return false;
                }
                // Overlapping memory is only legal between two aliasable
                // resources; otherwise the range must stay private.
                if (range.origin_resource != std::numeric_limits<std::uint32_t>::max() &&
                    !(range.origin_aliasable && requirement.aliasable)) {
                    return false;
                }
                const std::uint64_t aligned = AlignUp(range.offset, requirement.alignment);
                return aligned + size <= range.offset + range.size;
            };

            std::uint32_t chosen = std::numeric_limits<std::uint32_t>::max();
            std::uint64_t chosen_aligned = 0;
            for (std::uint32_t i = 0; i < ranges.size(); ++i) {
                if (!reusable(ranges[i])) {
                    continue;
                }
                if (chosen == std::numeric_limits<std::uint32_t>::max() ||
                    ranges[i].size < ranges[chosen].size) {
                    chosen = i;
                    chosen_aligned = AlignUp(ranges[i].offset, requirement.alignment);
                }
            }

            std::uint64_t offset = 0;
            if (chosen != std::numeric_limits<std::uint32_t>::max()) {
                const ReusableRange block = ranges[chosen];

                if (block.origin_resource != std::numeric_limits<std::uint32_t>::max()) {
                    plan.aliases.push_back(PlannedAlias{
                        .aliased_resource = index,
                        .after_resource = block.origin_resource,
                        .heap_key = heap_key,
                        .pass_index = requirement.first_pass,
                    });
                }

                // Left sliver, if alignment pushed the placement forward.
                std::vector<ReusableRange> remainders;
                if (chosen_aligned > block.offset) {
                    remainders.push_back(ReusableRange{
                        .offset = block.offset,
                        .size = chosen_aligned - block.offset,
                        .free_after_pass = block.free_after_pass,
                        .origin_resource = block.origin_resource,
                        .origin_aliasable = block.origin_aliasable,
                    });
                }
                const std::uint64_t block_end = block.offset + block.size;
                if (chosen_aligned + size < block_end) {
                    remainders.push_back(ReusableRange{
                        .offset = chosen_aligned + size,
                        .size = block_end - (chosen_aligned + size),
                        .free_after_pass = block.free_after_pass,
                        .origin_resource = block.origin_resource,
                        .origin_aliasable = block.origin_aliasable,
                    });
                }

                ranges.erase(ranges.begin() + static_cast<std::ptrdiff_t>(chosen));
                ranges.insert(ranges.end(), remainders.begin(), remainders.end());
                offset = chosen_aligned;
            } else {
                offset = AlignUp(copy_end, requirement.alignment);
                copy_end = offset + size;
            }

            ranges.push_back(ReusableRange{
                .offset = offset,
                .size = size,
                .free_after_pass = requirement.last_pass,
                .origin_resource = index,
                .origin_aliasable = requirement.aliasable,
            });
            offsets.emplace_back(index, offset);
        }

        // The per-frame stride must itself be aligned to the largest alignment
        // in the heap: each slot's offset is `slot * stride + base_offset`, and
        // base_offset is only aligned for its own resource. If the stride were
        // the unaligned copy_end, slot >= 1 could land an image on an offset
        // that violates vkBindImageMemory's alignment requirement.
        std::uint64_t max_alignment = 1;
        for (const std::uint32_t index : group) {
            max_alignment = std::max<std::uint64_t>(max_alignment, requirements[index].alignment);
        }
        const std::uint64_t stride = AlignUp(copy_end, max_alignment);
        plan.heap_copy_sizes.emplace_back(heap_key, stride);

        for (const auto& [resource_index, base_offset] : offsets) {
            const auto& requirement = requirements[resource_index];
            for (std::uint32_t slot = 0; slot < plan.frames_in_flight; ++slot) {
                plan.placements.push_back(TransientPlacement{
                    .resource_index = resource_index,
                    .fif_slot = slot,
                    .heap_key = heap_key,
                    .offset = static_cast<std::uint64_t>(slot) * stride + base_offset,
                    .size = std::max<std::uint64_t>(requirement.size, 1),
                });
            }
        }
    }

    return plan;
}

// ── bufferImageGranularity rule ──
//
// A linear (buffer) and a non-linear (optimal-tiled image) allocation must not
// share one bufferImageGranularity page. `PadForLinearity` advances the cursor
// to the next page when linearity changes, and `SharesGranularityPage` is the
// predicate the heap/debug path validates with.
[[nodiscard]] inline std::uint64_t PadForLinearity(std::uint64_t cursor,
                                                   bool previous_is_linear,
                                                   bool is_linear,
                                                   std::uint64_t granularity) {
    if (granularity <= 1 || previous_is_linear == is_linear) {
        return cursor;
    }
    return detail::AlignUp(cursor, granularity);
}

[[nodiscard]] inline bool SharesGranularityPage(std::uint64_t a_offset, std::uint64_t a_size,
                                                std::uint64_t b_offset, std::uint64_t b_size,
                                                std::uint64_t granularity) {
    if (granularity <= 1 || a_size == 0 || b_size == 0) {
        return false;
    }
    const std::uint64_t a_first = a_offset / granularity;
    const std::uint64_t a_last = (a_offset + a_size - 1) / granularity;
    const std::uint64_t b_first = b_offset / granularity;
    const std::uint64_t b_last = (b_offset + b_size - 1) / granularity;
    return a_first <= b_last && b_first <= a_last;
}

// ── Runtime allocator ──
//
// Realizes a TransientPlacementPlan on the device: one device-memory block per
// heap big enough for every FIF copy, with images bound at their planned
// offsets and buffers exposed as sub-ranges of one buffer. The planner owns the
// layout (including alias reuse); this class only binds to it, so TLSF-style
// best-fit is deliberately not re-applied here.
//
// Rebuilds (Sync) retire the previous generation and free it only once the frame
// that used it is GPU-complete (`IVulkanBootstrap::IsFrameComplete`), so a
// reconfiguration never frees memory the GPU is still reading.
class TransientAllocator {
public:
    struct ImageSpec {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        vk::Format format = vk::Format::eUndefined;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment;
        vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
        vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    struct BufferSpec {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
        vk::MemoryPropertyFlags memory = vk::MemoryPropertyFlagBits::eDeviceLocal;
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    struct Desc {
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        TransientRequirements requirements{};
        bool is_image = false;
        ImageSpec image{};
        BufferSpec buffer{};
        // NOLINTEND(misc-non-private-member-variables-in-classes)
    };

    TransientAllocator() = default;
    ~TransientAllocator();

    TransientAllocator(const TransientAllocator&) = delete;
    TransientAllocator& operator=(const TransientAllocator&) = delete;
    TransientAllocator(TransientAllocator&&) noexcept = default;
    TransientAllocator& operator=(TransientAllocator&&) noexcept = default;

    bool Initialize(VulkanBackend::Vulkan::IVulkanBootstrap& backend, const std::string& debug_name = "transients");
    void Shutdown();

    // Queue families transients are shared with. More than one family switches
    // transient images/buffers to concurrent sharing so a pass on another queue
    // (e.g. async compute) can access them without ownership transfers.
    void SetQueueFamilies(std::span<const std::uint32_t> families);

    // Rebuilds the plan/allocations if the descriptors changed. `current_frame`
    // is the frame currently being submitted, used to gate deferred frees.
    void Sync(const std::vector<Desc>& descs, std::uint32_t frames_in_flight, std::uint32_t current_frame);

    // Frees retired generations whose last using frame has completed.
    void CollectGarbage(std::uint32_t current_frame);

    [[nodiscard]] vk::Image GetImage(std::uint32_t resource_index, std::uint32_t fif_slot) const;
    [[nodiscard]] vk::ImageView GetImageView(std::uint32_t resource_index, std::uint32_t fif_slot) const;
    [[nodiscard]] bool GetBuffer(std::uint32_t resource_index, std::uint32_t fif_slot,
                                 vk::Buffer& out_buffer, vk::DeviceSize& out_offset, vk::DeviceSize& out_size) const;

    [[nodiscard]] const TransientPlacementPlan& GetPlan() const { return plan_; }
    [[nodiscard]] bool IsInitialized() const { return backend_ != nullptr; }

private:
    struct OwnedImage {
        std::unique_ptr<vk::raii::Image> image;
        std::unique_ptr<vk::raii::ImageView> view;
        std::unique_ptr<vk::raii::DeviceMemory> dedicated_memory;
    };

    struct Heap {
        std::uint32_t heap_key = 0;
        std::uint64_t total_size = 0;
        std::unique_ptr<vk::raii::DeviceMemory> memory;
        std::unique_ptr<vk::raii::Buffer> buffer;
    };

    struct Generation {
        std::vector<Heap> heaps;
        std::vector<OwnedImage> images;
        // resource_index -> image indices, one per FIF slot (empty for buffers).
        std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> image_lookup;
        std::vector<TransientPlacement> placements;
        TransientPlacementPlan plan{};
        std::uint64_t hash = 0;
        std::uint32_t retire_frame = 0;
    };

    [[nodiscard]] std::uint64_t HashDescs(const std::vector<Desc>& descs, std::uint32_t frames_in_flight) const;
    [[nodiscard]] bool BuildGeneration(const std::vector<Desc>& descs, std::uint32_t frames_in_flight, Generation& out);
    void ReleaseGeneration(Generation& generation);

    VulkanBackend::Vulkan::IVulkanBootstrap* backend_ = nullptr;
    std::string debug_name_ = "transients";
    std::vector<std::uint32_t> queue_families_{};
    TransientPlacementPlan plan_{};
    std::unique_ptr<Generation> current_{};
    std::vector<std::unique_ptr<Generation>> retired_{};
};

}  // namespace VulkanEngine::GpuResources
