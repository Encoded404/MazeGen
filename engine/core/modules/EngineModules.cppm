// Engine modules: the lifecycle seam for startup/shutdown choreography
// (rung 2 of engine modularization).
//
// An engine module is a feature with a lifecycle, not a class hierarchy:
//
//   Configure(GameConfig&)  — declare defaults the game may override
//   Startup(ModuleContext&) — construct subsystems, register systems/hooks
//   Shutdown(ModuleContext&)- teardown, reverse order of startup
//
// The engine's own preset (GameEngine::Setup) registers its subsystems through
// the exact same EngineModules surface a composed game uses, so the direct
// mode *degenerates into* the composed mode: with only the preset registered
// the resolved order is the old hardcoded choreography. Physics is the
// reference migration — its five-editor wiring (Setup block, Shutdown block,
// context member, accessor, export imports) collapsed into one module whose
// startup registers the slot systems and OnAdd/OnRemove hooks it already
// registered.
//
// Identity is never a string (same rules as SystemSchedule/FrameSchedule):
// registered modules can bind a ModuleSlot enumerator (engine-owned
// identities); anything else gets an opaque ModuleId handle back, and
// constraints reference either. Resolution runs at registration, is
// deterministic, and a bad constraint never leaves a half-registered module.
//
// Stages are the lifecycle's own structure, resolved at different points:
// a module registers for `Stage::Startup` (run by GameEngine::Setup),
// `Stage::Scene` (run once device-side render resources exist, after
// InitRenderer's core work — where scene building belongs), or both are
// independent registration points, not one ordering decision. Ordering
// constraints are only resolvable among modules of the same stage.

module;

#include <cassert>

export module VulkanEngine.Modules.EngineModules;

import std;

import VulkanEngine.Application;
import VulkanEngine.EngineContext;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.ECS.SystemSchedule;
import VulkanEngine.ECS.FrameSchedule;

export import VulkanEngine.Application;
export import VulkanEngine.EngineContext;
export import VulkanEngine.ECS.SystemSchedule;
export import VulkanEngine.ECS.FrameSchedule;

export namespace VulkanEngine::Modules {

/// The lifecycle points a module can hook. Each stage is started once, in the
/// resolved order of the modules registered for it. Shutdown walks the
/// complete start history in exact reverse, so an ordering decision made
/// between two stages cannot be violated by teardown.
enum class Stage : std::uint8_t {
    // NOLINTBEGIN(readability-identifier-naming)
    Startup = 0,  // GameEngine::Setup, after config resolution
    Scene = 1,    // after renderer init: scene building, resources, cameras
    // NOLINTEND(readability-identifier-naming)
};

[[nodiscard]] constexpr std::string_view StageName(Stage stage) noexcept {
    switch (stage) {
        case Stage::Startup: return "startup";
        case Stage::Scene:   return "scene";
    }
    return "unknown";
}

/// Engine-owned module identities that the default preset names. A composed
/// game orders its own modules against these with AddAfter / AddBefore.
enum class ModuleSlot : std::uint16_t {
    None = 0,          // not bound to a slot
    Device,            // the device bootstrap (heaps, bindless, text core...)
    Physics,           // the Jolt world and its ECS frame contract
    ResourceHooks,     // engine-owned resource release hooks
    PhysicalCamera,    // optional webcam subsystem lifecycle + event routing
    FramePipeline,     // the rendered-frame phase seeds
    Count,
};

[[nodiscard]] constexpr std::string_view ModuleSlotName(ModuleSlot slot) noexcept {
    switch (slot) {
        case ModuleSlot::Device:         return "engine.device";
        case ModuleSlot::Physics:        return "engine.physics";
        case ModuleSlot::ResourceHooks:  return "engine.resource_hooks";
        case ModuleSlot::PhysicalCamera: return "engine.physical_camera";
        case ModuleSlot::FramePipeline:  return "engine.frame_pipeline";
        case ModuleSlot::None:           return "<none>";
        case ModuleSlot::Count:          break;
    }
    return "<unknown>";
}

/// Opaque handle to a registered module: instance tag, registration index,
/// registry generation. Generation-checked like EntityId.
struct ModuleId {
    std::uint32_t owner = 0;
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool IsValid() const noexcept { return owner != 0 && generation != 0; }

    friend bool operator==(const ModuleId&, const ModuleId&) = default;
};

/// A reference to an already-registered module: an engine slot or a handle.
class ModuleRef {
public:
    constexpr ModuleRef() noexcept = default;
    // NOLINTNEXTLINE(google-explicit-constructor) — the implicit conversions are the point.
    constexpr ModuleRef(ModuleSlot slot) noexcept : slot_(slot) {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr ModuleRef(ModuleId handle) noexcept : handle_(handle) {}

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return slot_ != ModuleSlot::None || handle_.IsValid();
    }
    [[nodiscard]] constexpr ModuleSlot Slot() const noexcept { return slot_; }
    [[nodiscard]] constexpr ModuleId Handle() const noexcept { return handle_; }

    friend bool operator==(const ModuleRef&, const ModuleRef&) = default;

private:
    ModuleSlot slot_ = ModuleSlot::None;
    ModuleId handle_{};
};

/// What a module's Configure/Startup/Shutdown receive. Deliberately narrow:
/// the application context being set up, the engine context that owns the
/// subsystems, the three registration surfaces, and the resolved GameConfig.
/// A module never receives the GameEngine itself.
struct ModuleContext {
    Application::ApplicationContext* app = nullptr;               // NOLINT(misc-non-private-member-variables-in-classes)
    EngineContext* engine = nullptr;                              // NOLINT(misc-non-private-member-variables-in-classes)
    ComponentRegistry* registry = nullptr;                        // NOLINT(misc-non-private-member-variables-in-classes)
    SystemSchedule* fixed_systems = nullptr;                      // NOLINT(misc-non-private-member-variables-in-classes)
    FrameSchedule* frame_systems = nullptr;                       // NOLINT(misc-non-private-member-variables-in-classes)
    GameConfig* config = nullptr;                                 // NOLINT(misc-non-private-member-variables-in-classes)

    [[nodiscard]] Application::ApplicationContext& App() const noexcept {
        assert(app != nullptr);
        return *app;
    }
    [[nodiscard]] EngineContext& Ctx() const noexcept {
        assert(engine != nullptr);
        return *engine;
    }
    [[nodiscard]] ComponentRegistry& Registry() const noexcept {
        assert(registry != nullptr);
        return *registry;
    }
    [[nodiscard]] SystemSchedule& FixedSystems() const noexcept {
        assert(fixed_systems != nullptr);
        return *fixed_systems;
    }
    [[nodiscard]] FrameSchedule& FrameSystems() const noexcept {
        assert(frame_systems != nullptr);
        return *frame_systems;
    }
    [[nodiscard]] GameConfig& Config() const noexcept {
        assert(config != nullptr);
        return *config;
    }
};

/// Registration descriptor. `label` is diagnostics only and must have static
/// storage duration (pass a string literal). The functions may be empty.
struct EngineModuleDesc {
    Stage stage = Stage::Startup;
    std::string_view label = {};          // diagnostics only; static duration
    ModuleSlot slot = ModuleSlot::None;   // binds an engine-owned identity
    std::function<void(GameConfig&)> configure;
    std::function<bool(ModuleContext&)> startup;
    std::function<void(ModuleContext&)> shutdown;
};

/// Ordered engine-module registry, one sequence resolved per registration and
/// one start history kept in exact reverse for teardown.
///
/// Ordering contract (identical rules to SystemSchedule, restricted to a
/// stage):
///   - Registration order is startup order until a constraint says otherwise.
///   - AddAfter/AddBefore constrain the module being registered against one
///     that is already registered *in the same stage*. A target in another
///     stage throws at registration time, and the registry is left exactly as
///     it was.
///   - Configure(GameConfig&) runs in the same resolved order as startup.
///   - Resolution is deterministic: among the modules whose constraints are
///     already satisfied, the earliest-registered starts first. Each
///     registration carries at most one constraint edge, so a cycle cannot be
///     stated; the resolver still asserts acyclicity rather than trusting it.
class EngineModules {
public:
    EngineModules() noexcept : owner_(NextOwner()) {}
    EngineModules(const EngineModules&) = delete;
    EngineModules& operator=(const EngineModules&) = delete;
    EngineModules(EngineModules&&) = delete;
    EngineModules& operator=(EngineModules&&) = delete;
    ~EngineModules() = default;

    /// Register a module (unconstrained; joins its stage's registration order).
    ModuleId Add(const EngineModuleDesc& desc) {
        return Register(desc, {}, false);
    }

    /// Register a module to start after an already-registered one.
    ModuleId AddAfter(ModuleRef after, const EngineModuleDesc& desc) {
        return Register(desc, after, false);
    }

    /// Register a module to start before an already-registered one.
    ModuleId AddBefore(ModuleRef before, const EngineModuleDesc& desc) {
        return Register(desc, before, true);
    }

    /// Bind an engine-owned slot identity to a module (for later constraints).
    /// The bound module keeps its stage; the slot names it for constraints.
    ModuleId BindSlot(ModuleRef existing, ModuleSlot slot) {
        const std::uint16_t index = Resolve(existing);
        if (slot == ModuleSlot::None ||
            static_cast<std::size_t>(slot) >= slots_.size()) {
            throw std::logic_error("EngineModules: invalid ModuleSlot value");
        }
        if (slots_[static_cast<std::size_t>(slot)] != 0) {
            throw std::logic_error("EngineModules: slot '" + std::string(ModuleSlotName(slot)) +
                                   "' is already registered");
        }
        slots_[static_cast<std::size_t>(slot)] = static_cast<std::uint32_t>(index + 1);
        entries_[index].slot = slot;
        try {
            RebuildOrder();
        } catch (...) {
            slots_[static_cast<std::size_t>(slot)] = 0;
            entries_[index].slot = ModuleSlot::None;
            RebuildOrder();
            throw;
        }
        return ModuleId{owner_, index, generation_};
    }

    /// Run every module's Configure in resolved stage order. Runs all stages'
    /// configures together: default-declaration precedes every override.
    void ConfigureAll(GameConfig& config) {
        for (const std::uint32_t index : order_) {
            if (entries_[index].configure) {
                entries_[index].configure(config);
            }
        }
    }

    /// Start the modules of one stage, in resolved order. A module whose
    /// startup returns false stops the stage; already-started modules remain
    /// started so ShutdownAll() can unwind them.
    /// Returns false when a startup failed.
    bool StartupAll(Stage stage, ModuleContext& mctx) {
        for (const std::uint32_t index : order_) {
            Entry& entry = entries_[index];
            if (entry.stage != stage || entry.started) continue;
            if (entry.startup && !entry.startup(mctx)) {
                // Mark started so ShutdownAll unwinds it together with
                // everything that came before: a module that allocated anything
                // before failing still gets its teardown, reverse-ordered.
                entry.started = true;
                started_.push_back(index);
                return false;
            }
            entry.started = true;
            started_.push_back(index);
        }
        return true;
    }

    /// Shut down every module that was started, in the exact reverse of the
    /// start order across all stages. Idempotent.
    void ShutdownAll(ModuleContext& mctx) {
        for (auto it = started_.rbegin(); it != started_.rend(); ++it) {
            Entry& entry = entries_[*it];
            if (entry.shutdown) entry.shutdown(mctx);
        }
        started_.clear();
    }

    /// Whether every stage has been shut down (or nothing was ever started).
    [[nodiscard]] bool AllStopped() const noexcept { return started_.empty(); }

    [[nodiscard]] std::size_t Size() const noexcept { return entries_.size(); }

    /// Drop every module (only meaningful before startup). Handles taken
    /// before the call are invalidated.
    void Clear() noexcept {
        entries_.clear();
        order_.clear();
        constraints_.clear();
        slots_.fill(0);
        started_.clear();
        if (++generation_ == 0) {
            generation_ = 1;
        }
    }

private:
    static constexpr std::uint32_t kNoModule = static_cast<std::uint32_t>(-1);
    static constexpr std::size_t kMaxModules = std::numeric_limits<std::uint16_t>::max();

    /// Instance tag for ModuleId, so a handle from another registry is rejected.
    [[nodiscard]] static std::uint32_t NextOwner() noexcept {
        static std::atomic<std::uint32_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    struct Entry {
        std::string_view label;       // diagnostics only; static duration
        Stage stage = Stage::Startup;
        ModuleSlot slot = ModuleSlot::None;
        std::function<void(GameConfig&)> configure;
        std::function<bool(ModuleContext&)> startup;
        std::function<void(ModuleContext&)> shutdown;
        bool started = false;
    };

    /// One ordering edge: `before` starts before `after` (registration indices).
    struct Edge {
        std::uint32_t before;
        std::uint32_t after;
    };

    ModuleId Register(const EngineModuleDesc& desc, ModuleRef target,
                      bool start_before_target) {
        const bool constrained = target.IsValid();
        std::uint32_t target_index = kNoModule;
        if (constrained) {
            target_index = Resolve(target);
            if (desc.stage != entries_[target_index].stage) {
                throw std::logic_error("EngineModules: constraint crosses a stage boundary (target "
                                       "'" + std::string(Describe(target_index)) + "' runs in stage '" +
                                       std::string(StageName(entries_[target_index].stage)) +
                                       "', registration asked for '" +
                                       std::string(StageName(desc.stage)) + "')");
            }
        }

        // Register: resolve constraint first, then append, then rebuild.
        const ModuleId id = Append(desc);
        if (constrained) {
            constraints_.push_back(start_before_target ? Edge{id.index, target_index}
                                                       : Edge{target_index, id.index});
        }
        try {
            RebuildOrder();
        } catch (...) {
            if (constrained) {
                constraints_.pop_back();
            }
            entries_.pop_back();
            RebuildOrder(); // the previous state is known-acyclic; cannot throw
            throw;
        }
        return id;
    }

    ModuleId Append(const EngineModuleDesc& desc) {
        if (entries_.size() >= kMaxModules) {
            throw std::length_error("EngineModules: too many modules registered");
        }
        const ModuleSlot slot = desc.slot;
        if (slot != ModuleSlot::None) {
            const auto slot_index = static_cast<std::size_t>(slot);
            if (slot_index >= slots_.size()) {
                throw std::logic_error("EngineModules: invalid ModuleSlot value");
            }
            if (slots_[slot_index] != 0) {
                throw std::logic_error("EngineModules: slot '" +
                                       std::string(ModuleSlotName(slot)) +
                                       "' is already registered");
            }
        }
        const std::uint32_t index = static_cast<std::uint32_t>(entries_.size());
        entries_.push_back(Entry{
            .label = desc.label,
            .stage = desc.stage,
            .slot = slot,
            .configure = desc.configure,
            .startup = desc.startup,
            .shutdown = desc.shutdown,
        });
        if (slot != ModuleSlot::None) {
            slots_[static_cast<std::size_t>(slot)] = index + 1;
        }
        return ModuleId{owner_, index, generation_};
    }

    [[nodiscard]] std::uint32_t Resolve(ModuleRef ref) const {
        if (ref.Slot() != ModuleSlot::None) {
            const auto slot_index = static_cast<std::size_t>(ref.Slot());
            const std::uint32_t bound = slot_index < slots_.size() ? slots_[slot_index] : 0;
            if (bound == 0) {
                throw std::out_of_range("EngineModules: constraint refers to slot '" +
                                        std::string(ModuleSlotName(ref.Slot())) +
                                        "', which is not registered yet");
            }
            return bound - 1;
        }
        const ModuleId handle = ref.Handle();
        if (!handle.IsValid() || handle.owner != owner_ || handle.generation != generation_ ||
            handle.index >= entries_.size()) {
            throw std::out_of_range("EngineModules: constraint refers to a module handle that is "
                                    "stale, foreign, or was never registered");
        }
        return handle.index;
    }

    /// Re-resolve the startup order: stage groups in stage order, each
    /// resolved by Kahn's algorithm over that stage's constraint edges (lowest
    /// free registration index first — the same deterministic tie-break the
    /// system schedule uses). Throws std::logic_error on a cycle, leaving the
    /// previous order intact.
    void RebuildOrder() {
        const std::size_t count = entries_.size();
        std::vector<std::uint32_t> resolved;
        resolved.reserve(count);

        std::vector<std::vector<std::uint32_t>> dependents(count);
        for (const Edge& edge : constraints_) {
            dependents[edge.before].push_back(edge.after);
        }
        std::vector<std::uint32_t> pending(count, 0);
        for (const Edge& edge : constraints_) {
            ++pending[edge.after];
        }

        std::vector<bool> emitted(count, false);
        const auto last = static_cast<std::uint32_t>(count);
        for (const auto stage_value : std::views::iota(0u, 2u)) {
            const Stage stage = static_cast<Stage>(stage_value);
            const std::size_t in_stage =
                static_cast<std::size_t>(std::ranges::count_if(entries_,
                    [stage](const Entry& entry) { return entry.stage == stage; }));
            std::size_t emitted_in_stage = 0;
            for (std::size_t step = 0; step < in_stage; ++step) {
                std::uint32_t pick = kNoModule;
                for (std::uint32_t i = 0; i < last; ++i) {
                    if (!emitted[i] && entries_[i].stage == stage && pending[i] == 0) {
                        pick = i;
                        break;
                    }
                }
                if (pick == kNoModule) break;
                emitted[pick] = true;
                ++emitted_in_stage;
                resolved.push_back(pick);
                for (const std::uint32_t dependent : dependents[pick]) {
                    --pending[dependent];
                }
            }
            if (emitted_in_stage != in_stage) {
                std::string unresolved;
                for (std::uint32_t i = 0; i < last; ++i) {
                    if (!emitted[i] && entries_[i].stage == stage) {
                        if (!unresolved.empty()) unresolved += ", ";
                        unresolved += Describe(i);
                    }
                }
                throw std::logic_error("EngineModules: ordering cycle among '" +
                                       std::string(StageName(stage)) +
                                       "' modules: " + unresolved);
            }
        }
        order_ = std::move(resolved);
    }

    /// Human-readable name for diagnostics: the label if there is one, else
    /// the module slot it is bound to, else its registration index.
    [[nodiscard]] std::string Describe(std::uint32_t index) const {
        const Entry& entry = entries_[index];
        if (!entry.label.empty()) return std::string(entry.label);
        for (std::size_t slot_index = 1; slot_index < slots_.size(); ++slot_index) {
            if (slots_[slot_index] == index + 1) {
                return std::string(ModuleSlotName(static_cast<ModuleSlot>(slot_index)));
            }
        }
        return "module[" + std::to_string(index) + "]";
    }

    std::vector<Entry> entries_;       // registration order
    std::vector<std::uint32_t> order_; // startup order, as registration indices
    std::vector<Edge> constraints_;
    std::vector<std::uint32_t> started_; // start history for reverse teardown
    // slot -> registration index + 1, so 0 means "not registered".
    std::array<std::uint32_t, static_cast<std::size_t>(ModuleSlot::Count)> slots_{};
    const std::uint32_t owner_;
    std::uint32_t generation_ = 1;
};

} // namespace VulkanEngine::Modules