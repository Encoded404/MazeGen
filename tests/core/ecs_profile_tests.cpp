// Device-free ECS schedule profiling tests.
//
// The schedules accept a non-owning VulkanShared::ISystemProfileSink; these
// tests pin the contract: one call per executed system, in execution order,
// with the right kind, the resolved label, and a non-negative elapsed time.
// A null sink must leave the existing ordering behaviour untouched.

#include <gtest/gtest.h>

import std;

import VulkanShared.SystemProfile;
import VulkanEngine.Application;
import VulkanEngine.EngineContext;
import VulkanEngine.ECS.FrameSchedule;
import VulkanEngine.ECS.SystemSchedule;

namespace {

using namespace VulkanEngine;
using VulkanShared::SystemProfileKind;

struct RecordedTiming {
    SystemProfileKind kind;
    std::string label;
    double ms = 0.0;
};

class RecordingSink final : public VulkanShared::ISystemProfileSink {
public:
    void AddSystemTiming(const SystemProfileKind kind, const std::string_view label,
                         const double milliseconds) override {
        calls.push_back({kind, std::string(label), milliseconds});
    }

    std::vector<RecordedTiming> calls;
};

} // namespace

TEST(EcsProfileTest, FrameScheduleReportsEverySystemInExecutionOrder) {
    EngineContext engine{};
    Application::ApplicationContext app{};
    FramePhaseContext fctx{&app, &engine};

    FrameSchedule frames;
    frames.Add(FramePhase::Bridge, [](FramePhaseContext&) {}, "game.first");
    frames.Add(FramePhase::Bridge, [](FramePhaseContext&) {}, "game.second");
    frames.Add(FramePhase::Record, [](FramePhaseContext&) {}, "game.third");

    RecordingSink sink;
    frames.Run(FramePhase::Bridge, fctx, &sink);

    ASSERT_EQ(sink.calls.size(), 2u);
    EXPECT_EQ(sink.calls[0].kind, SystemProfileKind::Frame);
    EXPECT_EQ(sink.calls[0].label, "game.first");
    EXPECT_EQ(sink.calls[1].kind, SystemProfileKind::Frame);
    EXPECT_EQ(sink.calls[1].label, "game.second");
    EXPECT_GE(sink.calls[0].ms, 0.0);
    EXPECT_GE(sink.calls[1].ms, 0.0);

    // Running another phase reports only that phase's systems.
    sink.calls.clear();
    frames.Run(FramePhase::Record, fctx, &sink);
    ASSERT_EQ(sink.calls.size(), 1u);
    EXPECT_EQ(sink.calls[0].label, "game.third");
}

TEST(EcsProfileTest, FrameScheduleFallsBackToSlotNameThenRegistrationIndex) {
    EngineContext engine{};
    Application::ApplicationContext app{};
    FramePhaseContext fctx{&app, &engine};

    FrameSchedule frames;
    // Slot-bound, unnamed: the slot's diagnostic spelling is the label.
    frames.Add(FramePhase::Record, FrameSlot::RenderSubmit, [](FramePhaseContext&) {});
    frames.Run(FramePhase::Record, fctx);

    RecordingSink sink;
    frames.Run(FramePhase::Record, fctx, &sink);
    ASSERT_EQ(sink.calls.size(), 1u);
    EXPECT_EQ(sink.calls[0].label, "frame.render_submit");

    FrameSchedule unnamed;
    unnamed.Add(FramePhase::Bridge, [](FramePhaseContext&) {});
    unnamed.Run(FramePhase::Bridge, fctx, &sink);
    ASSERT_EQ(sink.calls.size(), 2u);
    EXPECT_EQ(sink.calls[1].label, "frame_system[0]");
}

TEST(EcsProfileTest, NullSinkLeavesFrameOrderingUnchanged) {
    EngineContext engine{};
    Application::ApplicationContext app{};
    FramePhaseContext fctx{&app, &engine};

    std::vector<std::string> order;
    FrameSchedule frames;
    frames.Add(FramePhase::Bridge, [&order](FramePhaseContext&) { order.push_back("a"); }, "a");
    frames.Add(FramePhase::Bridge, [&order](FramePhaseContext&) { order.push_back("b"); }, "b");

    frames.Run(FramePhase::Bridge, fctx);
    EXPECT_EQ(order, (std::vector<std::string>{"a", "b"}));
}

TEST(EcsProfileTest, SystemScheduleReportsEverySystemInExecutionOrder) {
    ComponentRegistry registry{};
    SystemSchedule systems;
    systems.Add([](ComponentRegistry&, float) {}, "physics.custom_a");
    systems.Add([](ComponentRegistry&, float) {}, "physics.custom_b");

    RecordingSink sink;
    systems.Run(registry, 0.016f, &sink);

    ASSERT_EQ(sink.calls.size(), 2u);
    EXPECT_EQ(sink.calls[0].kind, SystemProfileKind::Component);
    EXPECT_EQ(sink.calls[0].label, "physics.custom_a");
    EXPECT_EQ(sink.calls[1].kind, SystemProfileKind::Component);
    EXPECT_EQ(sink.calls[1].label, "physics.custom_b");
    EXPECT_GE(sink.calls[0].ms, 0.0);
}

TEST(EcsProfileTest, SystemScheduleFallsBackToSlotNameThenRegistrationIndex) {
    ComponentRegistry registry{};
    SystemSchedule systems;
    systems.Add(SystemSlot::PhysicsSimulate, [](ComponentRegistry&, float) {});

    RecordingSink sink;
    systems.Run(registry, 0.016f, &sink);
    ASSERT_EQ(sink.calls.size(), 1u);
    EXPECT_EQ(sink.calls[0].label, "physics.simulate");

    SystemSchedule unnamed;
    unnamed.Add([](ComponentRegistry&, float) {});
    unnamed.Run(registry, 0.016f, &sink);
    ASSERT_EQ(sink.calls.size(), 2u);
    EXPECT_EQ(sink.calls[1].label, "system[0]");
}

TEST(EcsProfileTest, SystemScheduleNullSinkStillRunsSystems) {
    ComponentRegistry registry{};
    int runs = 0;
    SystemSchedule systems;
    systems.Add([&runs](ComponentRegistry&, float) { ++runs; }, "count");

    systems.Run(registry, 0.016f);
    EXPECT_EQ(runs, 1);
}
