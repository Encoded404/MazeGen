#include <gtest/gtest.h>
#include <logging/logging_macros.hpp>  // LOGIFACE_LOG macro only — safe with import

import std;
import logiface;
import test_logging;

import VulkanEngine.ECS.ComponentContext;
import VulkanEngine.ECS.ComponentRegistry;
import VulkanEngine.Input;

namespace {

// Object component: derives Component, so it is stored as a stable object and
// receives OnAttach/Update.
class RegistryProbeComponent : public VulkanEngine::Component {
public:
    [[nodiscard]] bool WasAttached() const noexcept { return attached_; }
    [[nodiscard]] bool WasUpdated() const noexcept { return updated_; }
    [[nodiscard]] float GetLastDeltaTime() const noexcept { return last_delta_time_; }

    void OnAttach(VulkanEngine::Entity& /*owner*/) override { attached_ = true; }

    void Update(float delta_time, const VulkanEngine::ComponentContext& context) override {
        updated_ = true;
        last_delta_time_ = delta_time;
        last_context_ = &context;
    }

    [[nodiscard]] const VulkanEngine::ComponentContext* GetLastContext() const noexcept {
        return last_context_;
    }

private:
    bool attached_ = false;
    bool updated_ = false;
    float last_delta_time_ = 0.0f;
    const VulkanEngine::ComponentContext* last_context_ = nullptr;
};

// Data components: plain structs, so they live in dense columns.
struct ProbeData {
    int value = 0;
};

struct OtherData {
    int tag = 0;
};

}  // namespace

TEST(ComponentSystemTest, TypeIdSystemReturnsStableIdsPerType) {
    TestLogging::InstallPerTestFileLogger();

    struct TypeA {};
    struct TypeB {};

    const auto id_a_1 = VulkanEngine::ComponentTypeIDSystem::GetTypeID<TypeA>();
    const auto id_a_2 = VulkanEngine::ComponentTypeIDSystem::GetTypeID<TypeA>();
    const auto id_b = VulkanEngine::ComponentTypeIDSystem::GetTypeID<TypeB>();

    EXPECT_EQ(id_a_1, id_a_2);
    EXPECT_NE(id_a_1, id_b);
}

TEST(ComponentSystemTest, ObjectComponentReceivesOnAttachAndUpdate) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    auto& entity = registry.CreateEntity();
    auto& component = registry.AddComponent<RegistryProbeComponent>(entity);

    EXPECT_TRUE(component.WasAttached());
    EXPECT_FALSE(component.WasUpdated());
    EXPECT_TRUE(entity.HasComponent<RegistryProbeComponent>());
    EXPECT_EQ(entity.GetComponent<RegistryProbeComponent>(), &component);

    VulkanEngine::Input::InputSystem granted{};
    VulkanEngine::ComponentContext context{};
    context.input = &granted;
    registry.UpdateAllComponentsAsync(0.25f, context);
    EXPECT_TRUE(component.WasUpdated());
    EXPECT_FLOAT_EQ(component.GetLastDeltaTime(), 0.25f);
    // The granted view is what the update received — service access by grant,
    // not by construction-site injection.
    EXPECT_EQ(component.GetLastContext()->input, context.input);
}

TEST(ComponentSystemTest, DuplicateAddDoesNotReplaceTheStoredComponent) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    auto& entity = registry.CreateEntity();
    auto& first = registry.AddComponent<ProbeData>(entity);
    first.value = 42;

    EXPECT_THROW(registry.AddComponent<ProbeData>(entity), std::logic_error);
    EXPECT_EQ(entity.GetComponent<ProbeData>(), &first);
    EXPECT_EQ(registry.Count<ProbeData>(), 1u);
}

TEST(ComponentSystemTest, DataComponentsIterateDenselyInInsertionOrder) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    for (int i = 0; i < 5; ++i) {
        auto& entity = registry.CreateEntity();
        registry.AddComponent<ProbeData>(entity).value = i;
    }
    EXPECT_EQ(registry.Count<ProbeData>(), 5u);

    int expected = 0;
    registry.ForEach<ProbeData>([&](ProbeData& data) {
        EXPECT_EQ(data.value, expected);
        ++expected;
    });
    EXPECT_EQ(expected, 5);
}

TEST(ComponentSystemTest, ViewVisitsOnlyEntitiesWithEveryComponent) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    auto& both = registry.CreateEntity();
    registry.AddComponent<ProbeData>(both).value = 7;
    registry.AddComponent<OtherData>(both).tag = 1;

    auto& only_probe = registry.CreateEntity();
    registry.AddComponent<ProbeData>(only_probe).value = 9;

    std::size_t visited = 0;
    registry.ForEach<ProbeData, OtherData>(
        [&](VulkanEngine::Entity& entity, ProbeData& probe, OtherData& other) {
            ++visited;
            EXPECT_EQ(&entity, &both);
            EXPECT_EQ(probe.value, 7);
            EXPECT_EQ(other.tag, 1);
        });
    EXPECT_EQ(visited, 1u);
}

TEST(ComponentSystemTest, RemoveComponentClearsTheMaskAndTheColumn) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    auto& entity = registry.CreateEntity();
    registry.AddComponent<ProbeData>(entity).value = 3;
    ASSERT_TRUE(entity.HasComponent<ProbeData>());

    registry.RemoveComponent<ProbeData>(entity);
    EXPECT_FALSE(entity.HasComponent<ProbeData>());
    EXPECT_EQ(entity.GetComponent<ProbeData>(), nullptr);
    EXPECT_EQ(registry.Count<ProbeData>(), 0u);
}

TEST(ComponentSystemTest, DestroyEntityRemovesEverythingAndRejectsTheStaleId) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    auto& entity = registry.CreateEntity();
    const auto id = entity.GetId();
    registry.AddComponent<ProbeData>(entity).value = 1;
    registry.AddComponent<RegistryProbeComponent>(entity);
    EXPECT_EQ(registry.EntityCount(), 1u);

    registry.DestroyEntity(entity);

    EXPECT_FALSE(entity.IsAlive());
    EXPECT_FALSE(registry.IsAlive(id));
    EXPECT_EQ(registry.TryGetEntity(id), nullptr);
    EXPECT_EQ(registry.Count<ProbeData>(), 0u);
    EXPECT_EQ(registry.Count<RegistryProbeComponent>(), 0u);
    EXPECT_EQ(registry.EntityCount(), 0u);
}

TEST(ComponentSystemTest, ReusedSlotBumpsGeneration) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    auto& first = registry.CreateEntity();
    const auto stale = first.GetId();
    registry.DestroyEntity(first);

    auto& second = registry.CreateEntity();
    EXPECT_EQ(second.GetId().index, stale.index);
    EXPECT_NE(second.GetId().generation, stale.generation);
    EXPECT_FALSE(registry.IsAlive(stale));
    EXPECT_TRUE(registry.IsAlive(second.GetId()));
}

TEST(ComponentSystemTest, StructuralChangesInsideForEachAreDeferred) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    auto& keep = registry.CreateEntity();
    registry.AddComponent<ProbeData>(keep).value = 1;
    auto& doomed = registry.CreateEntity();
    registry.AddComponent<ProbeData>(doomed).value = 2;
    const auto doomed_id = doomed.GetId();

    registry.ForEach<ProbeData>([&](VulkanEngine::Entity& entity, ProbeData&) {
        if (entity.GetId() == doomed_id) {
            registry.DestroyEntity(entity);
        }
    });

    // Still present during iteration; applied at the phase boundary.
    EXPECT_EQ(registry.Count<ProbeData>(), 2u);
    registry.ApplyStructuralChanges();
    EXPECT_EQ(registry.Count<ProbeData>(), 1u);
    EXPECT_FALSE(registry.IsAlive(doomed_id));
}

TEST(ComponentSystemTest, EngineHooksFireOnAddAndRemove) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    int added = 0;
    int removed = 0;
    registry.OnAdd<ProbeData>([&](VulkanEngine::Entity&, ProbeData&) { ++added; });
    registry.OnRemove<ProbeData>([&](VulkanEngine::Entity&, ProbeData&) { ++removed; });

    auto& entity = registry.CreateEntity();
    registry.AddComponent<ProbeData>(entity);
    EXPECT_EQ(added, 1);

    registry.RemoveComponent<ProbeData>(entity);
    EXPECT_EQ(removed, 1);

    // Destroying an entity fires the remove hook even though it never calls
    // RemoveComponent explicitly.
    registry.AddComponent<ProbeData>(entity);
    registry.DestroyEntity(entity);
    EXPECT_EQ(added, 2);
    EXPECT_EQ(removed, 2);
}

TEST(ComponentSystemTest, ClearRemovesEveryEntityAndComponent) {
    TestLogging::InstallPerTestFileLogger();

    VulkanEngine::ComponentRegistry registry;
    for (int i = 0; i < 3; ++i) {
        auto& entity = registry.CreateEntity();
        registry.AddComponent<ProbeData>(entity);
        registry.AddComponent<RegistryProbeComponent>(entity);
    }

    registry.Clear();
    EXPECT_EQ(registry.EntityCount(), 0u);
    EXPECT_EQ(registry.Count<ProbeData>(), 0u);
    EXPECT_EQ(registry.Count<RegistryProbeComponent>(), 0u);
}
