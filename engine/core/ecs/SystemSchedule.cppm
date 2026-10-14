// Ordered data-system schedule for the hybrid ECS.
//
// Object components drive themselves through Component::Update. Data
// components do not: systems read them through registry views. Without an
// explicit order, that second path had no home in the frame. This schedule is
// deliberately minimal — an ordered list that runs once per fixed step, which is
// enough to make system order a decision instead of an accident.
//
// Identity is never a string:
//   - An engine-owned system that the frame contract names registers against a
//     SystemSlot enumerator. The enumerator *is* the identity, so a rename or a
//     typo is a compile error, and engine and game code agree on the spelling
//     without sharing a string.
//   - A system that a module or a game registers for itself gets an opaque
//     SystemId back (registration index + schedule generation, EntityId-style),
//     and constraints reference that handle.
//   - `label` is diagnostics only: it appears in failure messages and never
//     participates in resolution. It must have static storage duration (pass a
//     string literal) because the schedule stores a non-owning view.

module;

export module VulkanEngine.ECS.SystemSchedule;

import std;

export import VulkanEngine.ECS.ComponentRegistry;

export namespace VulkanEngine {

/// Engine-owned system identities that the fixed-step frame contract names.
enum class SystemSlot : std::uint16_t {
    None = 0,           // not bound to a slot
    PhysicsInput,       // gameplay writes -> body velocities, kinematic targets
    PhysicsSimulate,    // one blocking PhysicsSystem::Update
    PhysicsWriteBack,   // body poses -> Transform
    Count,
};

/// Diagnostic spelling of a slot. Identity is the enumerator; this is what
/// failure messages print.
[[nodiscard]] constexpr std::string_view SystemSlotName(SystemSlot slot) noexcept {
    switch (slot) {
        case SystemSlot::PhysicsInput:     return "physics.input";
        case SystemSlot::PhysicsSimulate:  return "physics.simulate";
        case SystemSlot::PhysicsWriteBack: return "physics.writeback";
        case SystemSlot::None:             return "<none>";
        case SystemSlot::Count:            break;
    }
    return "<unknown>";
}

/// Opaque handle to a registered system: schedule instance tag, registration
/// index, and schedule generation. Generation-checked like EntityId, so a handle
/// taken before Clear() — or one taken from a different schedule — cannot
/// silently resolve to another system.
struct SystemId {
    std::uint32_t owner = 0;
    std::uint16_t index = 0;
    std::uint16_t generation = 0;

    [[nodiscard]] constexpr bool IsValid() const noexcept { return owner != 0 && generation != 0; }

    friend bool operator==(const SystemId&, const SystemId&) = default;
};

/// A reference to an already-registered system: an engine slot or a handle.
/// Constraints take this type, so a mistyped name is not expressible.
class SystemRef {
public:
    constexpr SystemRef() noexcept = default;
    // NOLINTNEXTLINE(google-explicit-constructor) — the implicit conversions are the point.
    constexpr SystemRef(SystemSlot slot) noexcept : slot_(slot) {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr SystemRef(SystemId handle) noexcept : handle_(handle) {}

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return slot_ != SystemSlot::None || handle_.IsValid();
    }
    [[nodiscard]] constexpr SystemSlot Slot() const noexcept { return slot_; }
    [[nodiscard]] constexpr SystemId Handle() const noexcept { return handle_; }

    friend bool operator==(const SystemRef&, const SystemRef&) = default;

private:
    SystemSlot slot_ = SystemSlot::None;
    SystemId handle_{};
};

/// Ordered list of data-component systems, resolved once per registration.
///
/// Ordering contract:
///   - Registration order is execution order until a constraint says otherwise.
///   - `AddAfter`/`AddBefore` constrain the system being registered against one
///     that is already registered. Anything else (an unregistered or duplicated
///     slot, a stale or foreign handle) throws at registration time, and the
///     schedule is left exactly as it was — no half-registered system.
///   - Resolution runs at registration, never per frame, and is deterministic:
///     among the systems whose constraints are already satisfied, the
///     earliest-registered one runs first. Unconstrained systems therefore keep
///     registration order.
///   - Each registration carries at most one constraint edge, so a cycle cannot
///     be stated; the resolver still asserts acyclicity rather than trusting it.
class SystemSchedule {
public:
    using SystemFn = std::function<void(ComponentRegistry& registry, float delta_time)>;

    /// One schedule per engine: handles only mean anything to their owner, so
    /// copying the registration list would make every handle ambiguous.
    SystemSchedule() noexcept : owner_(NextOwner()) {}
    SystemSchedule(const SystemSchedule&) = delete;
    SystemSchedule& operator=(const SystemSchedule&) = delete;
    SystemSchedule(SystemSchedule&&) = delete;
    SystemSchedule& operator=(SystemSchedule&&) = delete;
    ~SystemSchedule() = default;

    /// Register an unconstrained system.
    SystemId Add(SystemFn fn, std::string_view label = {}) {
        return Register(std::move(fn), SystemSlot::None, label, {}, false);
    }

    /// Register a system bound to an engine-owned slot.
    SystemId Add(SystemSlot slot, SystemFn fn, std::string_view label = {}) {
        return Register(std::move(fn), slot, label, {}, false);
    }

    /// Register `fn` to run after an already-registered system.
    SystemId AddAfter(SystemRef after, SystemFn fn, std::string_view label = {}) {
        return Register(std::move(fn), SystemSlot::None, label, after, false);
    }

    /// Register `fn` to run after an already-registered system, slot-bound.
    SystemId AddAfter(SystemRef after, SystemSlot slot, SystemFn fn, std::string_view label = {}) {
        return Register(std::move(fn), slot, label, after, false);
    }

    /// Register `fn` to run before an already-registered system.
    SystemId AddBefore(SystemRef before, SystemFn fn, std::string_view label = {}) {
        return Register(std::move(fn), SystemSlot::None, label, before, true);
    }

    /// Register `fn` to run before an already-registered system, slot-bound.
    SystemId AddBefore(SystemRef before, SystemSlot slot, SystemFn fn, std::string_view label = {}) {
        return Register(std::move(fn), slot, label, before, true);
    }

    void Run(ComponentRegistry& registry, float delta_time) const {
        for (const std::uint16_t index : order_) {
            entries_[index].fn(registry, delta_time);
        }
    }

    [[nodiscard]] std::size_t Size() const noexcept { return entries_.size(); }

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

    /// Instance tag for SystemId, so a handle from another schedule is rejected
    /// instead of matching a same-index system there.
    [[nodiscard]] static std::uint32_t NextOwner() noexcept {
        static std::atomic<std::uint32_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    struct Entry {
        SystemFn fn;
        std::string_view label; // diagnostics only; static storage duration
    };

    /// One ordering edge: `before` runs before `after` (registration indices).
    struct Edge {
        std::uint16_t before;
        std::uint16_t after;
    };

    SystemId Register(SystemFn fn, SystemSlot slot, std::string_view label,
                      SystemRef target, bool run_before_target) {
        // Resolve the constraint first: a bad target must not leave a
        // half-registered system behind.
        const bool constrained = target.IsValid();
        const std::uint16_t target_index = constrained ? Resolve(target) : kNoSystem;

        const SystemId id = Append(std::move(fn), slot, label);
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
            if (slot != SystemSlot::None) {
                slots_[static_cast<std::size_t>(slot)] = 0;
            }
            entries_.pop_back();
            RebuildOrder(); // the previous state is known-acyclic; cannot throw
            throw;
        }
        return id;
    }

    SystemId Append(SystemFn fn, SystemSlot slot, std::string_view label) {
        if (entries_.size() >= kMaxSystems) {
            throw std::length_error("SystemSchedule: too many systems registered");
        }
        const std::uint16_t index = static_cast<std::uint16_t>(entries_.size());
        if (slot != SystemSlot::None) {
            const auto slot_index = static_cast<std::size_t>(slot);
            if (slot_index >= slots_.size()) {
                throw std::logic_error("SystemSchedule: invalid SystemSlot value");
            }
            if (slots_[slot_index] != 0) {
                throw std::logic_error("SystemSchedule: slot '" + std::string(SystemSlotName(slot)) +
                                       "' is already registered");
            }
        }
        entries_.push_back(Entry{std::move(fn), label});
        // Bind only after the entry exists, so a throwing push_back cannot leave
        // a slot pointing at a system that was never registered.
        if (slot != SystemSlot::None) {
            slots_[static_cast<std::size_t>(slot)] = static_cast<std::uint16_t>(index + 1);
        }
        return SystemId{owner_, index, generation_};
    }

    [[nodiscard]] std::uint16_t Resolve(SystemRef ref) const {
        if (ref.Slot() != SystemSlot::None) {
            const auto slot_index = static_cast<std::size_t>(ref.Slot());
            const std::uint16_t bound = slot_index < slots_.size() ? slots_[slot_index] : 0;
            if (bound == 0) {
                throw std::out_of_range("SystemSchedule: constraint refers to slot '" +
                                        std::string(SystemSlotName(ref.Slot())) +
                                        "', which is not registered yet");
            }
            return static_cast<std::uint16_t>(bound - 1);
        }
        const SystemId handle = ref.Handle();
        if (!handle.IsValid() || handle.owner != owner_ || handle.generation != generation_ ||
            handle.index >= entries_.size()) {
            throw std::out_of_range("SystemSchedule: constraint refers to a system handle that is "
                                    "stale, foreign, or was never registered");
        }
        return handle.index;
    }

    /// Re-resolve the execution order. Throws std::logic_error on a cycle,
    /// leaving the previous order intact.
    void RebuildOrder() {
        const std::size_t count = entries_.size();
        std::vector<std::uint16_t> resolved(count);
        if (constraints_.empty()) {
            for (std::size_t i = 0; i < count; ++i) {
                resolved[i] = static_cast<std::uint16_t>(i);
            }
            order_ = std::move(resolved);
            return;
        }

        std::vector<std::uint32_t> pending(count, 0);
        std::vector<std::vector<std::uint16_t>> dependents(count);
        for (const Edge& edge : constraints_) {
            ++pending[edge.after];
            dependents[edge.before].push_back(edge.after);
        }

        // Kahn's algorithm, always taking the lowest registration index that is
        // currently free: deterministic, and it reduces to registration order
        // when nothing is constrained.
        std::vector<bool> emitted(count, false);
        const auto last = static_cast<std::uint16_t>(count);
        std::size_t next = 0;
        for (std::size_t step = 0; step < count; ++step) {
            std::uint16_t pick = kNoSystem;
            for (std::uint16_t i = 0; i < last; ++i) {
                if (!emitted[i] && pending[i] == 0) {
                    pick = i;
                    break;
                }
            }
            if (pick == kNoSystem) {
                break;
            }
            emitted[pick] = true;
            resolved[next++] = pick;
            for (const std::uint16_t dependent : dependents[pick]) {
                --pending[dependent];
            }
        }
        if (next != count) {
            std::string unresolved;
            for (std::uint16_t i = 0; i < last; ++i) {
                if (!emitted[i]) {
                    if (!unresolved.empty()) {
                        unresolved += ", ";
                    }
                    unresolved += Describe(i);
                }
            }
            throw std::logic_error("SystemSchedule: ordering cycle among systems: " + unresolved);
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
                return std::string(SystemSlotName(static_cast<SystemSlot>(slot_index)));
            }
        }
        return "system[" + std::to_string(index) + "]";
    }

    std::vector<Entry> entries_;       // registration order
    std::vector<std::uint16_t> order_; // execution order, as registration indices
    std::vector<Edge> constraints_;
    // slot -> registration index + 1, so 0 means "not registered".
    std::array<std::uint16_t, static_cast<std::size_t>(SystemSlot::Count)> slots_{};
    const std::uint32_t owner_;
    std::uint16_t generation_ = 1;
};

} // namespace VulkanEngine
