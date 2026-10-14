#include <gtest/gtest.h>

import std;

import VulkanEngine.ECS.SystemSchedule;

namespace {

using VulkanEngine::ComponentRegistry;
using VulkanEngine::SystemId;
using VulkanEngine::SystemSchedule;
using VulkanEngine::SystemSlot;
using VulkanEngine::SystemSlotName;

struct Counter {
    int value = 0;
};

SystemSchedule::SystemFn Recorder(std::vector<std::string>& order, std::string name) {
    return [&order, name = std::move(name)](ComponentRegistry&, float) { order.push_back(name); };
}

} // namespace

TEST(SystemScheduleTest, RunsSystemsInRegistrationOrder) {
    ComponentRegistry registry;
    auto& entity = registry.CreateEntity();
    registry.AddComponent<Counter>(entity);

    std::vector<std::string> order;
    SystemSchedule schedule;
    schedule.Add(Recorder(order, "first"), "first");
    schedule.Add(Recorder(order, "second"), "second");
    schedule.Add([&](ComponentRegistry& reg, float) {
        order.push_back("third");
        if (auto* counter = reg.GetComponent<Counter>(entity)) counter->value += 1;
    }, "third");

    ASSERT_EQ(schedule.Size(), 3u);
    schedule.Run(registry, 0.016f);

    const std::vector<std::string> expected{"first", "second", "third"};
    EXPECT_EQ(order, expected);
    EXPECT_EQ(registry.GetComponent<Counter>(entity)->value, 1);
}

TEST(SystemScheduleTest, EmptyScheduleRunsNothing) {
    ComponentRegistry registry;
    SystemSchedule schedule;
    EXPECT_EQ(schedule.Size(), 0u);
    schedule.Run(registry, 0.016f); // must not crash
    SUCCEED();
}

TEST(SystemScheduleTest, ClearRemovesEverySystem) {
    SystemSchedule schedule;
    schedule.Add([](ComponentRegistry&, float) {}, "a");
    ASSERT_EQ(schedule.Size(), 1u);

    schedule.Clear();
    EXPECT_EQ(schedule.Size(), 0u);
}

TEST(SystemScheduleTest, AddAfterHonorsTheConstraintAndKeepsEarlierSystemsFirst) {
    ComponentRegistry registry;
    std::vector<std::string> order;
    SystemSchedule schedule;

    const SystemId a = schedule.Add(Recorder(order, "a"), "a");
    schedule.Add(Recorder(order, "b"), "b");
    // Only "c" is constrained; "b" was registered earlier and is free, so it
    // still runs first — constrained systems move, unconstrained ones do not.
    schedule.AddAfter(a, Recorder(order, "c"), "c");

    schedule.Run(registry, 0.0f);
    const std::vector<std::string> expected{"a", "b", "c"};
    EXPECT_EQ(order, expected);
}

TEST(SystemScheduleTest, AddBeforeMovesTheTargetBehindTheNewSystem) {
    ComponentRegistry registry;
    std::vector<std::string> order;
    SystemSchedule schedule;

    const SystemId a = schedule.Add(Recorder(order, "a"), "a");
    schedule.Add(Recorder(order, "b"), "b");
    // "c" must precede "a", so "a" slides behind it while "b" keeps its place.
    schedule.AddBefore(a, Recorder(order, "c"), "c");

    schedule.Run(registry, 0.0f);
    const std::vector<std::string> expected{"b", "c", "a"};
    EXPECT_EQ(order, expected);
}

TEST(SystemScheduleTest, SlotIdentityOrdersTheFrameContract) {
    ComponentRegistry registry;
    std::vector<std::string> order;
    SystemSchedule schedule;

    schedule.Add(SystemSlot::PhysicsInput, Recorder(order, "physics.input"));
    schedule.AddAfter(SystemSlot::PhysicsInput, SystemSlot::PhysicsSimulate,
                      Recorder(order, "physics.simulate"));
    schedule.AddAfter(SystemSlot::PhysicsSimulate, SystemSlot::PhysicsWriteBack,
                      Recorder(order, "physics.writeback"));
    // A game system constrains itself against an engine slot by name-free identity.
    schedule.AddAfter(SystemSlot::PhysicsWriteBack, Recorder(order, "game.camera"), "game.camera");

    schedule.Run(registry, 0.0f);
    const std::vector<std::string> expected{
        "physics.input", "physics.simulate", "physics.writeback", "game.camera"};
    EXPECT_EQ(order, expected);
}

TEST(SystemScheduleTest, ConstrainedScheduleSatisfiesEveryConstraint) {
    ComponentRegistry registry;
    std::vector<std::string> order;
    SystemSchedule schedule;

    // Deliberately mixes forward and backward references. Derivation of the
    // expected order (lowest free registration index wins each step):
    //   s1 before s0, s0 before s2, s3 before s2, s1 before s4
    //   -> s1, s0, s3, s2, s4
    const SystemId s0 = schedule.Add(Recorder(order, "s0"), "s0");
    const SystemId s1 = schedule.AddBefore(s0, Recorder(order, "s1"), "s1");
    const SystemId s2 = schedule.AddAfter(s0, Recorder(order, "s2"), "s2");
    schedule.AddBefore(s2, Recorder(order, "s3"), "s3");
    schedule.AddAfter(s1, Recorder(order, "s4"), "s4");

    schedule.Run(registry, 0.0f);
    const std::vector<std::string> expected{"s1", "s0", "s3", "s2", "s4"};
    EXPECT_EQ(order, expected);
}

TEST(SystemScheduleTest, UnknownSlotTargetThrowsAndRegistersNothing) {
    ComponentRegistry registry;
    SystemSchedule schedule;

    bool threw = false;
    try {
        schedule.AddAfter(SystemSlot::PhysicsSimulate, [](ComponentRegistry&, float) {});
    } catch (const std::out_of_range& error) {
        threw = true;
        // The diagnostic reads the slot's human-readable name; identity stays
        // the enumerator.
        EXPECT_NE(std::string(error.what()).find("physics.simulate"), std::string::npos);
    }
    EXPECT_TRUE(threw);
    EXPECT_EQ(schedule.Size(), 0u); // no half-registered system
    schedule.Run(registry, 0.0f);
}

TEST(SystemScheduleTest, StaleAndForeignHandlesAreRejected) {
    SystemSchedule schedule;
    const SystemId stale = schedule.Add([](ComponentRegistry&, float) {});
    schedule.Clear();
    EXPECT_THROW(schedule.AddAfter(stale, [](ComponentRegistry&, float) {}), std::out_of_range);
    EXPECT_EQ(schedule.Size(), 0u);

    // A handle from another schedule occupies the same index there; it must not
    // resolve against this one.
    SystemSchedule other;
    const SystemId foreign = other.Add([](ComponentRegistry&, float) {});
    schedule.Add([](ComponentRegistry&, float) {}); // occupy index 0 here too
    EXPECT_THROW(schedule.AddAfter(foreign, [](ComponentRegistry&, float) {}), std::out_of_range);
    EXPECT_EQ(schedule.Size(), 1u);
}

TEST(SystemScheduleTest, SlotCannotBeRegisteredTwice) {
    SystemSchedule schedule;
    schedule.Add(SystemSlot::PhysicsInput, [](ComponentRegistry&, float) {});
    EXPECT_THROW(schedule.Add(SystemSlot::PhysicsInput, [](ComponentRegistry&, float) {}),
                 std::logic_error);
    EXPECT_EQ(schedule.Size(), 1u);
}

TEST(SystemScheduleTest, SlotNamesMatchTheFrameContract) {
    EXPECT_EQ(SystemSlotName(SystemSlot::PhysicsInput), "physics.input");
    EXPECT_EQ(SystemSlotName(SystemSlot::PhysicsSimulate), "physics.simulate");
    EXPECT_EQ(SystemSlotName(SystemSlot::PhysicsWriteBack), "physics.writeback");
}

TEST(SystemScheduleTest, DestroyOutsideIterationAppliesImmediately) {
    ComponentRegistry registry;
    auto& doomed = registry.CreateEntity();
    registry.AddComponent<Counter>(doomed);
    const auto doomed_id = doomed.GetId();

    SystemSchedule schedule;
    schedule.Add([&](ComponentRegistry& reg, float) {
        if (auto* entity = reg.TryGetEntity(doomed_id)) reg.DestroyEntity(*entity);
    }, "destroy-doomed");
    schedule.Run(registry, 0.0f);

    // Nothing was iterating, so the queue fast-path applied it at once.
    EXPECT_EQ(registry.Count<Counter>(), 0u);
    EXPECT_FALSE(registry.IsAlive(doomed_id));
}

TEST(SystemScheduleTest, DestroyInsideAViewIsDeferredUntilThePhaseBoundary) {
    ComponentRegistry registry;
    auto& keep = registry.CreateEntity();
    registry.AddComponent<Counter>(keep);
    auto& doomed = registry.CreateEntity();
    registry.AddComponent<Counter>(doomed);
    const auto doomed_id = doomed.GetId();

    SystemSchedule schedule;
    schedule.Add([&](ComponentRegistry& reg, float) {
        reg.ForEach<Counter>([&](VulkanEngine::Entity& entity, Counter&) {
            if (entity.GetId() == doomed_id) reg.DestroyEntity(entity);
        });
    }, "destroy-doomed");
    schedule.Run(registry, 0.0f);

    // Still present until the fixed-update boundary applies the queue.
    EXPECT_EQ(registry.Count<Counter>(), 2u);
    registry.ApplyStructuralChanges();
    EXPECT_EQ(registry.Count<Counter>(), 1u);
    EXPECT_FALSE(registry.IsAlive(doomed_id));
}
