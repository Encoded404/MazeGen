// Hybrid ECS: object components and data components in one registry.
//
// The contract, in one place:
//
//   * A type that derives from `Component` is an OBJECT component. It is stored
//     as a stable heap object, may hold resources and virtual behavior, and a
//     `T*` to it stays valid for the entity's life.
//   * Any other type is a DATA component. It is stored in a dense contiguous
//     column and addressed through a sparse set. A `T*`/`T&` to it is
//     TRANSIENT: valid until the next structural change (add/remove/destroy).
//     Store an EntityId and re-fetch instead of caching the pointer.
//
// Both kinds live on the same Entity and are queried through the same
// registry; only storage and addressability differ.
//
// Identity: `EntityId` is the persistent handle (generation-checked across slot
// reuse). `Entity&` is valid until the next structural change.
//
// Threading: structural changes and queries are single-threaded and never
// overlap. `GetComponent`/`HasComponent` are lock-free reads; the mutating APIs
// and `ForEach` take the registry mutex. Object-component `Update` runs in
// parallel over stable pointers. No structural change may occur while a query
// or update is in flight.
//
// Structural changes: `RemoveComponent` and `DestroyEntity` called from inside
// `ForEach` are deferred and applied by `ApplyStructuralChanges()`. `CreateEntity`
// and `AddComponent` are immediate and assert if called during iteration —
// collect what you want to spawn, then spawn after the loop.

module;

#include <cassert>

export module VulkanEngine.ECS.ComponentRegistry;

import std;
import std.compat;

export import VulkanEngine.ECS.Entity;

import VulkanShared.Jobs;

export namespace VulkanEngine {

class Entity;
class ComponentRegistry;

// ── Component base (object components only) ──
//
// Data components are plain structs and do not derive from this. Keep behavior
// and resource ownership here; keep bulk data out of it.
class Component {
public:
    virtual ~Component() = default;

    // Called once, immediately after the component is attached and its owner is
    // set. Replaces the old lazy Initialize().
    virtual void OnAttach(Entity& /*owner*/) {}
    // Called immediately before the component is removed or its entity is
    // destroyed, while the component is still intact.
    virtual void OnDetach(Entity& /*owner*/) {}
    // Per-step behavior for object components. Data components are driven by
    // systems instead.
    virtual void Update(float /*delta_time*/) {}

    void SetOwner(Entity* entity) noexcept { owner_ = entity; }
    [[nodiscard]] Entity* GetOwner() const noexcept { return owner_; }
    [[nodiscard]] EntityId GetOwnerId() const noexcept;

private:
    Entity* owner_ = nullptr;
};

// ── Entity ──
class Entity {
public:
    Entity() = default;

    [[nodiscard]] EntityId GetId() const noexcept { return id_; }
    [[nodiscard]] bool IsAlive() const noexcept { return alive_; }
    [[nodiscard]] ComponentMask GetMask() const noexcept { return mask_; }
    [[nodiscard]] ComponentRegistry& GetRegistry() const noexcept { return *registry_; }
    [[nodiscard]] ComponentRegistry* GetRegistryPtr() const noexcept { return registry_; }

    template<typename T>
    [[nodiscard]] T* GetComponent();
    template<typename T>
    [[nodiscard]] const T* GetComponent() const;

    template<typename T>
    [[nodiscard]] bool HasComponent() const {
        const std::size_t type_id = ComponentTypeIDSystem::GetTypeID<T>();
        return type_id < kMaxComponentTypes && mask_.test(type_id);
    }

private:
    friend class ComponentRegistry;

    void Activate(std::uint32_t index, ComponentRegistry* registry) noexcept {
        id_ = EntityId{index, generation_};
        registry_ = registry;
        alive_ = true;
        mask_.reset();
    }

    void MarkDead() noexcept {
        alive_ = false;
        mask_.reset();
        if (++generation_ == 0) generation_ = 1; // never reuse generation 0
    }

    EntityId id_{};
    ComponentRegistry* registry_ = nullptr;
    ComponentMask mask_{};
    bool alive_ = false;
    std::uint32_t generation_ = 1;
};

// ── Component pools ──
class IComponentPool {
public:
    virtual ~IComponentPool() = default;

    [[nodiscard]] virtual bool Has(const Entity& entity) const = 0;
    virtual bool Remove(Entity& entity) = 0;
    virtual void Clear() = 0;
    [[nodiscard]] virtual std::size_t Count() const = 0;
    virtual void CollectUpdatables(std::vector<Component*>& out) = 0;
    virtual void ForEachEntity(const std::function<void(Entity&)>& fn) = 0;
};

// Dense column for data components, stable heap objects for object components.
template<typename T>
class ComponentPool final : public IComponentPool {
public:
    static constexpr bool kObject = std::is_base_of_v<Component, T>;

    template<typename... Args>
    T& Emplace(Entity& owner, Args&&... args) {
        if (Has(owner)) {
            throw std::logic_error("Entity already owns this component type");
        }
        const std::uint32_t index = owner.GetId().index;
        EnsureSparse(index);

        if constexpr (kObject) {
            values_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
            values_.back()->SetOwner(&owner);
        } else {
            values_.emplace_back(std::forward<Args>(args)...);
        }
        owners_.push_back(&owner);
        sparse_[index] = static_cast<std::uint32_t>(values_.size()); // row 0 -> 1

        T& component = *At(values_.size() - 1);
        if constexpr (kObject) {
            component.OnAttach(owner);
        } else if constexpr (requires(T& value, Entity& entity) { value.OnAttach(entity); }) {
            component.OnAttach(owner);
        }
        for (auto& hook : on_add_) hook(owner, component);
        return component;
    }

    [[nodiscard]] T* Find(const Entity& entity) {
        const std::uint32_t row = FindRow(entity);
        return row == kNoRow ? nullptr : At(row);
    }
    [[nodiscard]] const T* Find(const Entity& entity) const {
        const std::uint32_t row = FindRow(entity);
        return row == kNoRow ? nullptr : At(row);
    }

    [[nodiscard]] bool Has(const Entity& entity) const override {
        return FindRow(entity) != kNoRow;
    }

    bool Remove(Entity& entity) override {
        const std::uint32_t row = FindRow(entity);
        if (row == kNoRow) return false;

        T& component = *At(row);
        if constexpr (kObject) {
            component.OnDetach(entity);
        } else if constexpr (requires(T& value, Entity& owner) { value.OnDetach(owner); }) {
            component.OnDetach(entity);
        }
        for (auto& hook : on_remove_) hook(entity, component);

        // Swap-delete: the moved-into-row owner gets its sparse index fixed up.
        const std::size_t last = values_.size() - 1;
        if (row != last) {
            std::swap(values_[row], values_[last]);
            std::swap(owners_[row], owners_[last]);
            sparse_[owners_[row]->GetId().index] = row + 1;
        }
        values_.pop_back();
        owners_.pop_back();
        sparse_[entity.GetId().index] = 0;
        return true;
    }

    void Clear() override {
        for (std::size_t i = 0; i < values_.size(); ++i) {
            T& component = *At(i);
            Entity& owner = *owners_[i];
            if constexpr (kObject) {
                component.OnDetach(owner);
            } else if constexpr (requires(T& value, Entity& entity) { value.OnDetach(entity); }) {
                component.OnDetach(owner);
            }
            for (auto& hook : on_remove_) hook(owner, component);
        }
        values_.clear();
        owners_.clear();
        sparse_.clear();
    }

    [[nodiscard]] std::size_t Count() const override { return values_.size(); }

    void CollectUpdatables(std::vector<Component*>& out) override {
        if constexpr (kObject) {
            for (auto& value : values_) out.push_back(value.get());
        }
    }

    void ForEachEntity(const std::function<void(Entity&)>& fn) override {
        for (Entity* owner : owners_) fn(*owner);
    }

    void AddOnAdd(std::function<void(Entity&, T&)> hook) { on_add_.push_back(std::move(hook)); }
    void AddOnRemove(std::function<void(Entity&, T&)> hook) { on_remove_.push_back(std::move(hook)); }

private:
    static constexpr std::uint32_t kNoRow = std::numeric_limits<std::uint32_t>::max();

    using Storage = std::conditional_t<kObject, std::vector<std::unique_ptr<T>>, std::vector<T>>;

    [[nodiscard]] T* At(std::size_t row) {
        if constexpr (kObject) return values_[row].get();
        else return &values_[row];
    }
    [[nodiscard]] const T* At(std::size_t row) const {
        if constexpr (kObject) return values_[row].get();
        else return &values_[row];
    }

    [[nodiscard]] std::uint32_t FindRow(const Entity& entity) const {
        const std::uint32_t index = entity.GetId().index;
        if (index >= sparse_.size()) return kNoRow;
        const std::uint32_t encoded = sparse_[index];
        return encoded == 0 ? kNoRow : encoded - 1;
    }

    void EnsureSparse(std::uint32_t index) {
        if (sparse_.size() <= index) sparse_.resize(static_cast<std::size_t>(index) + 1, 0);
    }

    Storage values_{};
    std::vector<Entity*> owners_{};
    std::vector<std::uint32_t> sparse_{};
    std::vector<std::function<void(Entity&, T&)>> on_add_{};
    std::vector<std::function<void(Entity&, T&)>> on_remove_{};
};

// ── Registry ──
class ComponentRegistry {
public:
    ComponentRegistry() = default;
    ~ComponentRegistry() { Clear(); }

    ComponentRegistry(const ComponentRegistry&) = delete;
    ComponentRegistry& operator=(const ComponentRegistry&) = delete;

    // ── Entities ──
    Entity& CreateEntity() {
        const std::scoped_lock lock(mutex_);
        std::uint32_t index = 0;
        if (!free_slots_.empty()) {
            index = free_slots_.back();
            free_slots_.pop_back();
        } else {
            index = static_cast<std::uint32_t>(slots_.size());
            slots_.push_back(std::make_unique<Entity>());
        }
        Entity& entity = *slots_[index];
        entity.Activate(index, this);
        return entity;
    }

    void DestroyEntity(Entity& entity) {
        if (iteration_depth_ > 0) {
            const EntityId id = entity.GetId();
            pending_.push_back([this, id] {
                if (Entity* target = TryGetEntity(id)) DestroyNow(*target);
            });
            return;
        }
        const std::scoped_lock lock(mutex_);
        DestroyNow(entity);
    }

    void DestroyEntity(EntityId id) {
        if (Entity* entity = TryGetEntity(id)) DestroyEntity(*entity);
    }

    [[nodiscard]] bool IsAlive(EntityId id) const {
        if (id.index >= slots_.size()) return false;
        const Entity& entity = *slots_[id.index];
        return entity.alive_ && entity.id_.generation == id.generation;
    }

    [[nodiscard]] std::size_t EntityCount() const {
        const std::scoped_lock lock(mutex_);
        return slots_.size() - free_slots_.size();
    }

    [[nodiscard]] Entity* TryGetEntity(EntityId id) {
        if (!IsAlive(id)) return nullptr;
        return slots_[id.index].get();
    }

    template<typename Fn>
    void ForEachEntity(Fn&& fn) {
        const std::scoped_lock lock(mutex_);
        for (auto& slot : slots_) {
            if (slot != nullptr && slot->alive_) fn(*slot);
        }
    }

    // ── Components ──
    template<typename T, typename... Args>
    T& AddComponent(Entity& owner, Args&&... args) {
        assert(iteration_depth_ == 0 &&
               "AddComponent is immediate; collect spawns and add them after iteration");
        const std::scoped_lock lock(mutex_);
        if (!owner.alive_) {
            throw std::logic_error("AddComponent on a destroyed entity");
        }
        const std::size_t type_id = ComponentTypeIDSystem::GetTypeID<T>();
        if (type_id >= kMaxComponentTypes) {
            throw std::logic_error("component type budget exceeded (" +
                                   std::to_string(kMaxComponentTypes) + " types)");
        }
        auto& pool = GetOrCreatePool<T>();
        T& component = pool.Emplace(owner, std::forward<Args>(args)...);
        owner.mask_.set(type_id);
        return component;
    }

    template<typename T>
    void RemoveComponent(Entity& entity) {
        if (iteration_depth_ > 0) {
            const EntityId id = entity.GetId();
            pending_.push_back([this, id] {
                if (Entity* target = TryGetEntity(id)) RemoveInternal<T>(*target);
            });
            return;
        }
        const std::scoped_lock lock(mutex_);
        RemoveInternal<T>(entity);
    }

    template<typename T>
    [[nodiscard]] T* GetComponent(Entity& entity) {
        if (!HasComponent<T>(entity)) return nullptr;
        auto* pool = GetPool<T>();
        return pool != nullptr ? pool->Find(entity) : nullptr;
    }

    template<typename T>
    [[nodiscard]] const T* GetComponent(const Entity& entity) const {
        if (!HasComponent<T>(entity)) return nullptr;
        const auto* pool = GetPool<T>();
        return pool != nullptr ? pool->Find(entity) : nullptr;
    }

    template<typename T>
    [[nodiscard]] bool HasComponent(const Entity& entity) const {
        const std::size_t type_id = ComponentTypeIDSystem::GetTypeID<T>();
        return type_id < kMaxComponentTypes && entity.mask_.test(type_id);
    }

    template<typename T>
    [[nodiscard]] std::size_t Count() const {
        const std::scoped_lock lock(mutex_);
        const auto* pool = GetPool<T>();
        return pool != nullptr ? pool->Count() : 0;
    }

    // ── Queries ──
    //
    // Visits every entity that has all of Ts, iterating the smallest matching
    // pool so empty intersections cost nothing. `fn` may take (Entity&, Ts&...)
    // or (Ts&...). Order is the driving pool's insertion order (deterministic).
    template<typename... Ts, typename Fn>
    void ForEach(Fn&& fn) {
        static_assert(sizeof...(Ts) > 0, "ForEach requires at least one component type");
        const std::scoped_lock lock(mutex_);
        ++iteration_depth_;
        struct DepthGuard {
            std::size_t& depth;
            ~DepthGuard() { --depth; }
        } guard{iteration_depth_};

        const std::array<IComponentPool*, sizeof...(Ts)> candidates{
            static_cast<IComponentPool*>(GetPool<Ts>())...};

        IComponentPool* driver = nullptr;
        for (IComponentPool* candidate : candidates) {
            if (candidate == nullptr) continue;
            if (driver == nullptr || candidate->Count() < driver->Count()) driver = candidate;
        }
        if (driver == nullptr || driver->Count() == 0) return;

        driver->ForEachEntity([&](Entity& entity) {
            if (!(HasComponent<Ts>(entity) && ...)) return;
            if constexpr (std::is_invocable_v<Fn&, Entity&, Ts&...>) {
                std::invoke(fn, entity, *GetComponent<Ts>(entity)...);
            } else {
                std::invoke(fn, *GetComponent<Ts>(entity)...);
            }
        });
    }

    // ── Engine-owned resource hooks ──
    template<typename T, typename Fn>
    void OnAdd(Fn&& fn) {
        const std::scoped_lock lock(mutex_);
        GetOrCreatePool<T>().AddOnAdd(std::forward<Fn>(fn));
    }

    template<typename T, typename Fn>
    void OnRemove(Fn&& fn) {
        const std::scoped_lock lock(mutex_);
        GetOrCreatePool<T>().AddOnRemove(std::forward<Fn>(fn));
    }

    // ── Object-component behavior ──
    void UpdateAllComponentsAsync(float delta_time) {
        std::vector<Component*> components;
        {
            const std::scoped_lock lock(mutex_);
            for (auto& pool : pools_) {
                if (pool != nullptr) pool->CollectUpdatables(components);
            }
        }
        if (components.empty()) return;

        VulkanShared::JobScheduler::Global().ParallelFor(components.size(),
            [delta_time, components = std::move(components)](const std::size_t index) mutable {
                components[index]->Update(delta_time);
            });
    }

    // ── Structural queue ──
    void ApplyStructuralChanges() {
        const std::scoped_lock lock(mutex_);
        std::vector<std::function<void()>> pending;
        pending.swap(pending_);
        for (auto& action : pending) action();
    }

    void Clear() {
        const std::scoped_lock lock(mutex_);
        pending_.clear();
        for (auto& slot : slots_) {
            if (slot != nullptr && slot->alive_) DestroyNow(*slot);
        }
        slots_.clear();
        free_slots_.clear();
        pools_.clear();
    }

private:
    template<typename T>
    [[nodiscard]] ComponentPool<T>* GetPool() {
        const std::size_t type_id = ComponentTypeIDSystem::GetTypeID<T>();
        if (type_id >= pools_.size() || pools_[type_id] == nullptr) return nullptr;
        return static_cast<ComponentPool<T>*>(pools_[type_id].get());
    }

    template<typename T>
    [[nodiscard]] const ComponentPool<T>* GetPool() const {
        const std::size_t type_id = ComponentTypeIDSystem::GetTypeID<T>();
        if (type_id >= pools_.size() || pools_[type_id] == nullptr) return nullptr;
        return static_cast<const ComponentPool<T>*>(pools_[type_id].get());
    }

    template<typename T>
    ComponentPool<T>& GetOrCreatePool() {
        const std::size_t type_id = ComponentTypeIDSystem::GetTypeID<T>();
        if (type_id >= kMaxComponentTypes) {
            throw std::logic_error("component type budget exceeded");
        }
        if (type_id >= pools_.size()) pools_.resize(type_id + 1);
        if (pools_[type_id] == nullptr) {
            pools_[type_id] = std::make_unique<ComponentPool<T>>();
        }
        return *static_cast<ComponentPool<T>*>(pools_[type_id].get());
    }

    void DestroyNow(Entity& entity) {
        if (!entity.alive_) return;
        for (auto& pool : pools_) {
            if (pool != nullptr) pool->Remove(entity);
        }
        entity.MarkDead();
        free_slots_.push_back(entity.id_.index);
    }

    template<typename T>
    void RemoveInternal(Entity& entity) {
        const std::size_t type_id = ComponentTypeIDSystem::GetTypeID<T>();
        if (type_id >= kMaxComponentTypes || !entity.mask_.test(type_id)) return;
        if (auto* pool = GetPool<T>()) pool->Remove(entity);
        entity.mask_.reset(type_id);
    }

    std::vector<std::unique_ptr<IComponentPool>> pools_{};
    std::vector<std::unique_ptr<Entity>> slots_{};
    std::vector<std::uint32_t> free_slots_{};
    std::vector<std::function<void()>> pending_{};
    std::size_t iteration_depth_ = 0;
    mutable std::mutex mutex_{};
};

// ── Out-of-line definitions that need the registry complete ──
inline EntityId Component::GetOwnerId() const noexcept {
    return owner_ != nullptr ? owner_->GetId() : EntityId{};
}

template<typename T>
T* Entity::GetComponent() {
    return registry_ != nullptr ? registry_->GetComponent<T>(*this) : nullptr;
}

template<typename T>
const T* Entity::GetComponent() const {
    return registry_ != nullptr ? registry_->GetComponent<T>(*this) : nullptr;
}

} // namespace VulkanEngine
