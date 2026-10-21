// Frame phases as schedules (rung 1 of engine modularization).
//
// GameEngine's per-rendered-frame pipeline used to be a private calling
// sequence. This schedule gives it the same shape the fixed step already has:
// named phases, each backed by a SystemSchedule-style ordered registry, so the
// frame contract is a tested artifact instead of comment paragraphs.
//
// Phases run in a fixed external order (the frame's own structure, not an
// ordering decision):
//
//   Bridge      — once per rendered frame, after the fixed steps, before
//                 recording: per-frame buffer indexing, interpolation pumps.
//   PreRecord   — drains and flushes that must land before the frame builds
//                 GPU commands: bindless drain, hot-reload pumps, residency,
//                 dirty materials, technique recompiles.
//   Record      — render submission itself, which is a phase participant,
//                 not the phase.
//   PostPresent — next-frame telemetry, per-frame cleanup.
//
// Identity is never a string (same rules as SystemSchedule):
//   - An engine-owned frame system registers against a FrameSlot enumerator;
//     the enumerator *is* the identity, so a rename is a compile error.
//   - A system a module or game registers for itself gets an opaque FrameId
//     back (instance tag + index + generation), and constraints reference that
//     handle. `label` is diagnostics only and must be a string literal.
//   - A constraint may only name a target in the same phase: a phase's order is
//     resolved inside the phase, because the phase sequence itself is fixed.
//     A cross-phase AddAfter is a mistake that is caught at registration.
//
// The deterministic tie-break is the fixed-step resolver's: registration order
// is execution order until a constraint says otherwise, so the engine's seed
// systems, registered in the order the old pipeline called them, reproduce the
// previous frame behavior exactly.

module;

#include <cassert>

export module VulkanEngine.ECS.FrameSchedule;

import std;

import VulkanEngine.Application;
import VulkanEngine.EngineContext;
import VulkanShared.SystemProfile;

export namespace VulkanEngine {

/// The fixed sequence of rendered-frame phases. The phase order is the frame's
/// structure; only the systems inside a phase are ordered by this schedule.
enum class FramePhase : std::uint16_t {
    None = 0,      // not bound to a phase
    Bridge,        // sim-to-render bridge (per-frame indexing, pumps)
    PreRecord,     // drains and flushes before the frame builds GPU commands
    Record,        // render submission
    PostPresent,   // after the frame is submitted
    Count,
};

/// Diagnostic spelling of a phase.
[[nodiscard]] constexpr std::string_view FramePhaseName(FramePhase phase) noexcept {
    switch (phase) {
        case FramePhase::Bridge:      return "bridge";
        case FramePhase::PreRecord:   return "pre-record";
        case FramePhase::Record:      return "record";
        case FramePhase::PostPresent: return "post-present";
        case FramePhase::None:        return "<none>";
        case FramePhase::Count:       break;
    }
    return "<unknown>";
}

/// Engine-owned frame-system identities that the default frame contract names.
/// A game hooks the same seam with AddAfter/AddBefore on one of these.
enum class FrameSlot : std::uint16_t {
    None = 0,             // not bound to a slot
    TextDisplayScale,     // platform content scale -> text system (runs even
                          // when no valid scene exists: UI still draws)
    MeshProcessFrame,     // per-frame mesh/render processing (FIF ring indexing)
    MeshEndFrame,         // mesh manager's per-frame retire
    TextureUploadPump,    // finished async uploads -> staged images
    BindlessDrain,        // deferred bindless descriptor ops
    TextureReloadPump,    // texture hot reload (builds with hot reload only)
    FontReloadPump,       // font hot reload (builds with hot reload only)
    ResidencyCollect,     // LRU eviction under the memory budget
    MaterialFlush,        // dirty material data -> GPU
    TechniquePoll,        // technique pipelines whose shaders changed
    PhysicalCameraPoll,   // webcam composite pipeline shaders (optional)
    RenderSubmit,         // the render submission (Record phase)
    Count,
};

/// Diagnostic spelling of a frame slot.
[[nodiscard]] constexpr std::string_view FrameSlotName(FrameSlot slot) noexcept {
    switch (slot) {
        case FrameSlot::TextDisplayScale:   return "frame.text_display_scale";
        case FrameSlot::MeshProcessFrame:   return "frame.mesh_process";
        case FrameSlot::MeshEndFrame:       return "frame.mesh_end";
        case FrameSlot::TextureUploadPump:  return "frame.texture_upload_pump";
        case FrameSlot::BindlessDrain:      return "frame.bindless_drain";
        case FrameSlot::TextureReloadPump:  return "frame.texture_reload_pump";
        case FrameSlot::FontReloadPump:     return "frame.font_reload_pump";
        case FrameSlot::ResidencyCollect:   return "frame.residency_collect";
        case FrameSlot::MaterialFlush:      return "frame.material_flush";
        case FrameSlot::TechniquePoll:      return "frame.technique_poll";
        case FrameSlot::PhysicalCameraPoll: return "frame.physical_camera_poll";
        case FrameSlot::RenderSubmit:       return "frame.render_submit";
        case FrameSlot::None:               return "<none>";
        case FrameSlot::Count:              break;
    }
    return "<unknown>";
}

/// What a frame system sees. Deliberately narrow: the frame state the loop is
/// driving and the engine context that owns the subsystems. A frame system
/// never receives the GameEngine itself.
struct FramePhaseContext {
    Application::ApplicationContext* app = nullptr;    // NOLINT(misc-non-private-member-variables-in-classes)
    EngineContext* engine = nullptr;                   // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] Application::ApplicationContext& App() const noexcept {
        assert(app != nullptr);
        return *app;
    }
    [[nodiscard]] EngineContext& Engine() const noexcept {
        assert(engine != nullptr);
        return *engine;
    }
};

/// Opaque handle to a registered frame system: instance tag, registration
/// index, schedule generation. Generation-checked like EntityId.
struct FrameId {
    std::uint32_t owner = 0;
    std::uint16_t index = 0;
    std::uint16_t generation = 0;

    [[nodiscard]] constexpr bool IsValid() const noexcept { return owner != 0 && generation != 0; }

    friend bool operator==(const FrameId&, const FrameId&) = default;
};

/// A reference to an already-registered frame system: an engine slot or a
/// handle. Constraints take this type, so a mistyped name is not expressible.
class FrameRef {
public:
    constexpr FrameRef() noexcept = default;
    // NOLINTNEXTLINE(google-explicit-constructor) — the implicit conversions are the point.
    constexpr FrameRef(FrameSlot slot) noexcept : slot_(slot) {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr FrameRef(FrameId handle) noexcept : handle_(handle) {}

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return slot_ != FrameSlot::None || handle_.IsValid();
    }
    [[nodiscard]] constexpr FrameSlot Slot() const noexcept { return slot_; }
    [[nodiscard]] constexpr FrameId Handle() const noexcept { return handle_; }

    friend bool operator==(const FrameRef&, const FrameRef&) = default;

private:
    FrameSlot slot_ = FrameSlot::None;
    FrameId handle_{};
};

/// Ordered frame-phase registry: one registration-time resolver, one phase's
/// systems per Run call.
///
/// Ordering contract (identical rules to SystemSchedule, restricted to a
/// phase):
///   - Registration order is execution order until a constraint says otherwise.
///   - AddAfter/AddBefore constrain the system being registered against one
///     that is already registered *in the same phase*. A target in another
///     phase throws at registration time, and the schedule is left exactly as
///     it was — no half-registered system, and never a cross-phase edge that
///     would let a constraint fight the fixed phase sequence.
///   - Resolution runs at registration, never per frame, and is deterministic:
///     among the systems whose constraints are already satisfied, the
///     earliest-registered one runs first (Kahn's algorithm taking the lowest
///     registration index currently free).
///   - Each registration carries at most one constraint edge, so a cycle
///     cannot be stated; the resolver still asserts acyclicity rather than
///     trusting it.
class FrameSchedule {
public:
    using FrameFn = std::function<void(FramePhaseContext&)>;

    /// One schedule per engine: handles only mean anything to their owner.
    FrameSchedule() noexcept : owner_(NextOwner()) {}
    FrameSchedule(const FrameSchedule&) = delete;
    FrameSchedule& operator=(const FrameSchedule&) = delete;
    FrameSchedule(FrameSchedule&&) = delete;
    FrameSchedule& operator=(FrameSchedule&&) = delete;
    ~FrameSchedule() = default;

    /// Register an unconstrained system into `phase`.
    FrameId Add(FramePhase phase, FrameFn fn, std::string_view label = {}) {
        return Register(std::move(fn), phase, FrameSlot::None, label, {}, false);
    }

    /// Register a system bound to an engine-owned slot identity, into `phase`.
    FrameId Add(FramePhase phase, FrameSlot slot, FrameFn fn, std::string_view label = {}) {
        return Register(std::move(fn), phase, slot, label, {}, false);
    }

    /// Register `fn` to run after an already-registered system. The new system
    /// joins the target's phase.
    FrameId AddAfter(FrameRef after, FrameFn fn, std::string_view label = {}) {
        return Register(std::move(fn), FramePhase::None, FrameSlot::None, label, after, false);
    }

    /// Register `fn` to run after an already-registered system, slot-bound.
    FrameId AddAfter(FrameRef after, FrameSlot slot, FrameFn fn, std::string_view label = {}) {
        return Register(std::move(fn), FramePhase::None, slot, label, after, false);
    }

    /// Register `fn` to run before an already-registered system.
    FrameId AddBefore(FrameRef before, FrameFn fn, std::string_view label = {}) {
        return Register(std::move(fn), FramePhase::None, FrameSlot::None, label, before, true);
    }

    /// Register `fn` to run before an already-registered system, slot-bound.
    FrameId AddBefore(FrameRef before, FrameSlot slot, FrameFn fn, std::string_view label = {}) {
        return Register(std::move(fn), FramePhase::None, slot, label, before, true);
    }

    /// Run one phase's systems in its resolved order. Phases with no systems
    /// cost nothing. A non-null `profile` receives one timing per executed
    /// system; a null one keeps the original branch-free fast path.
    void Run(FramePhase phase, FramePhaseContext& fctx,
             VulkanShared::ISystemProfileSink* profile = nullptr) const {
        if (profile == nullptr) {
            for (const std::uint16_t index : order_) {
                if (entries_[index].phase == phase) {
                    entries_[index].fn(fctx);
                }
            }
            return;
        }
        for (const std::uint16_t index : order_) {
            if (entries_[index].phase != phase) {
                continue;
            }
            // Resolve the label before the clock starts, so name lookup is not
            // included in the measurement and evaluation order cannot matter.
            const std::string label = Describe(index);
            const auto start = std::chrono::steady_clock::now();
            entries_[index].fn(fctx);
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - start)
                                          .count();
            profile->AddSystemTiming(VulkanShared::SystemProfileKind::Frame, label, elapsed_ms);
        }
    }

    [[nodiscard]] std::size_t Size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::size_t PhaseSize(FramePhase phase) const noexcept {
        std::size_t count = 0;
        for (const Entry& entry : entries_) {
            if (entry.phase == phase) ++count;
        }
        return count;
    }

    /// Drop every system. Handles taken before the call are invalidated.
    void Clear() noexcept {
        entries_.clear();
        order_.clear();
        constraints_.clear();
        slots_.fill(0);
        if (++generation_ == 0) {
            generation_ = 1;
        }
    }

private:
    static constexpr std::uint16_t kNoSystem = static_cast<std::uint16_t>(-1);
    static constexpr std::size_t kMaxSystems = 65535;

    /// Instance tag for FrameId, so a handle from another schedule is rejected
    /// instead of matching a same-index system there.
    [[nodiscard]] static std::uint32_t NextOwner() noexcept {
        static std::atomic<std::uint32_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    struct Entry {
        FrameFn fn;
        std::string_view label; // diagnostics only; static storage duration
        FramePhase phase = FramePhase::None;
    };

    /// One ordering edge: `before` runs before `after` (registration indices).
    struct Edge {
        std::uint16_t before;
        std::uint16_t after;
    };

    FrameId Register(FrameFn fn, FramePhase phase, FrameSlot slot, std::string_view label,
                     FrameRef target, bool run_before_target) {
        // Resolve the constraint first: a bad target must not leave a
        // half-registered system behind.
        const bool constrained = target.IsValid();
        std::uint16_t target_index = kNoSystem;
        if (constrained) {
            target_index = Resolve(target);
            // The new system joins the target's phase, unless an explicit slot
            // registration declared a different one — catch that mistake here.
            if (phase == FramePhase::None) {
                phase = entries_[target_index].phase;
            } else if (entries_[target_index].phase != phase) {
                throw std::logic_error("FrameSchedule: constraint crosses a phase boundary (target "
                                       "'" + std::string(Describe(target_index)) + "' runs in phase '" +
                                       std::string(FramePhaseName(entries_[target_index].phase)) +
                                       "', registration asked for '" + std::string(FramePhaseName(phase)) + "')");
            }
        }
        if (phase == FramePhase::None || static_cast<std::size_t>(phase) >= static_cast<std::size_t>(FramePhase::Count)) {
            throw std::logic_error("FrameSchedule: registration needs a known target or a valid phase");
        }

        const FrameId id = Append(std::move(fn), phase, slot, label);
        if (constrained) {
            constraints_.push_back(run_before_target ? Edge{id.index, target_index}
                                                     : Edge{target_index, id.index});
        }
        try {
            RebuildOrder();
        } catch (...) {
            if (constrained) {
                constraints_.pop_back();
            }
            if (slot != FrameSlot::None) {
                slots_[static_cast<std::size_t>(slot)] = 0;
            }
            entries_.pop_back();
            RebuildOrder(); // the previous state is known-acyclic; cannot throw
            throw;
        }
        return id;
    }

    FrameId Append(FrameFn fn, FramePhase phase, FrameSlot slot, std::string_view label) {
        if (entries_.size() >= kMaxSystems) {
            throw std::length_error("FrameSchedule: too many systems registered");
        }
        const std::uint16_t index = static_cast<std::uint16_t>(entries_.size());
        if (slot != FrameSlot::None) {
            const auto slot_index = static_cast<std::size_t>(slot);
            if (slot_index >= slots_.size()) {
                throw std::logic_error("FrameSchedule: invalid FrameSlot value");
            }
            if (slots_[slot_index] != 0) {
                throw std::logic_error("FrameSchedule: slot '" + std::string(FrameSlotName(slot)) +
                                       "' is already registered");
            }
        }
        entries_.push_back(Entry{std::move(fn), label, phase});
        // Bind only after the entry exists, so a throwing push_back cannot leave
        // a slot pointing at a system that was never registered.
        if (slot != FrameSlot::None) {
            slots_[static_cast<std::size_t>(slot)] = static_cast<std::uint16_t>(index + 1);
        }
        return FrameId{owner_, index, generation_};
    }

    [[nodiscard]] std::uint16_t Resolve(FrameRef ref) const {
        if (ref.Slot() != FrameSlot::None) {
            const auto slot_index = static_cast<std::size_t>(ref.Slot());
            const std::uint16_t bound = slot_index < slots_.size() ? slots_[slot_index] : 0;
            if (bound == 0) {
                throw std::out_of_range("FrameSchedule: constraint refers to slot '" +
                                        std::string(FrameSlotName(ref.Slot())) +
                                        "', which is not registered yet");
            }
            return static_cast<std::uint16_t>(bound - 1);
        }
        const FrameId handle = ref.Handle();
        if (!handle.IsValid() || handle.owner != owner_ || handle.generation != generation_ ||
            handle.index >= entries_.size()) {
            throw std::out_of_range("FrameSchedule: constraint refers to a system handle that is "
                                    "stale, foreign, or was never registered");
        }
        return handle.index;
    }

    /// Re-resolve the execution order: phase groups in phase order, each
    /// resolved by Kahn's algorithm over that phase's constraint edges. Throws
    /// std::logic_error on a cycle, leaving the previous order intact.
    void RebuildOrder() {
        const std::size_t count = entries_.size();
        std::vector<std::uint16_t> resolved;
        resolved.reserve(count);

        std::vector<std::uint32_t> pending(count, 0);
        std::vector<std::vector<std::uint16_t>> dependents(count);
        for (const Edge& edge : constraints_) {
            ++pending[edge.after];
            dependents[edge.before].push_back(edge.after);
        }

        std::vector<bool> emitted(count, false);
        const auto last = static_cast<std::uint16_t>(count);
        for (const std::size_t phase_value :
             std::views::iota(std::size_t{1}, static_cast<std::size_t>(FramePhase::Count))) {
            const FramePhase phase = static_cast<FramePhase>(phase_value);
            std::size_t emitted_in_phase = 0;
            const std::size_t in_phase = static_cast<std::size_t>(std::ranges::count_if(
                entries_, [phase](const Entry& entry) { return entry.phase == phase; }));
            for (std::size_t step = 0; step < in_phase; ++step) {
                // Always take the lowest registration index that is currently
                // free: deterministic, and it reduces to registration order
                // when nothing is constrained. The phase guard keeps the scan
                // inside this phase's group.
                std::uint16_t pick = kNoSystem;
                for (std::uint16_t i = 0; i < last; ++i) {
                    if (!emitted[i] && entries_[i].phase == phase && pending[i] == 0) {
                        pick = i;
                        break;
                    }
                }
                if (pick == kNoSystem) break;
                emitted[pick] = true;
                ++emitted_in_phase;
                resolved.push_back(pick);
                for (const std::uint16_t dependent : dependents[pick]) {
                    --pending[dependent];
                }
            }
            if (emitted_in_phase != in_phase) {
                std::string unresolved;
                for (std::uint16_t i = 0; i < last; ++i) {
                    if (!emitted[i] && entries_[i].phase == phase) {
                        if (!unresolved.empty()) {
                            unresolved += ", ";
                        }
                        unresolved += Describe(i);
                    }
                }
                throw std::logic_error("FrameSchedule: ordering cycle among '" +
                                       std::string(FramePhaseName(phase)) +
                                       "' systems: " + unresolved);
            }
        }
        order_ = std::move(resolved);
    }

    /// Human-readable name for diagnostics: the label if there is one, else the
    /// slot this system is bound to, else its registration index.
    [[nodiscard]] std::string Describe(std::uint16_t index) const {
        const Entry& entry = entries_[index];
        if (!entry.label.empty()) {
            return std::string(entry.label);
        }
        for (std::size_t slot_index = 1; slot_index < slots_.size(); ++slot_index) {
            if (slots_[slot_index] == static_cast<std::uint16_t>(index + 1)) {
                return std::string(FrameSlotName(static_cast<FrameSlot>(slot_index)));
            }
        }
        return "frame_system[" + std::to_string(index) + "]";
    }

    std::vector<Entry> entries_;       // registration order
    std::vector<std::uint16_t> order_; // execution order, as registration indices
    std::vector<Edge> constraints_;
    // slot -> registration index + 1, so 0 means "not registered".
    std::array<std::uint16_t, static_cast<std::size_t>(FrameSlot::Count)> slots_{};
    const std::uint32_t owner_;
    std::uint16_t generation_ = 1;
};

} // namespace VulkanEngine