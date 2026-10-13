#include <gtest/gtest.h>

import std;
import std.compat;

import Examples.InfiniteRunner.Balance;

namespace {

using Examples::InfiniteRunner::BalanceConfig;

// Golden fingerprint of the v1 default ruleset, computed with the documented
// FNV-1a-64 contract (version, then every field in declaration order, floats
// folded via their bit pattern). Update deliberately when kVersion changes.
constexpr std::uint64_t kGoldenV1 = 0xEE1A8498415DBD96ULL;

TEST(BalanceConfigTest, DefaultRulesetFingerprintIsStable) {
    EXPECT_EQ(BalanceConfig{}.Hash(), kGoldenV1);
}

TEST(BalanceConfigTest, FingerprintIsDeterministic) {
    EXPECT_EQ(BalanceConfig{}.Hash(), BalanceConfig{}.Hash());
}

// INTENTIONAL: every gameplay-affecting field must participate in the
// fingerprint; a field that silently stopped being hashed would merge two
// different rulesets onto one leaderboard. This per-field sweep is a deliberate
// tripwire, not a mirror of the hasher — do not fold it away as redundant with
// the golden value above. A golden hash can only catch a change to *some*
// field; this catches a change to *each* one.
TEST(BalanceConfigTest, EveryFieldChangesTheFingerprint) {
    const BalanceConfig base{};
    const std::uint64_t baseline = base.Hash();

    const auto changed = [&](auto mutate) {
        BalanceConfig c = base;
        mutate(c);
        return c.Hash();
    };

    EXPECT_NE(changed([](BalanceConfig& c) { c.corridor_half += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_height += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_depth += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.min_block_width += 0.001f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.player_size += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.player_speed += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_speed += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_spawn_z -= 1.0f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_initial_spawn_z -= 1.0f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_recycle_z += 1.0f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_spacing_pow_scaling += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_count += 1; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_hole_min += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_hole_max += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_hole_size_pow_scaling += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_hole_placement_min += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.wall_hole_placement_max += 0.01f; }), baseline);
    EXPECT_NE(changed([](BalanceConfig& c) { c.difficulty_scaling_divider += 0.1f; }), baseline);
}

// The getter documents wall spacing as the span each wall occupies: `wall_count`
// spacings must cover the corridor from spawn to recycle exactly. Assert the
// invariant the getter documents rather than re-deriving the value, so a
// reformulation that keeps the corridor covered still passes.
TEST(BalanceConfigTest, WallSpacingIsDerivedFromFields) {
    const BalanceConfig c{};
    EXPECT_FLOAT_EQ(c.WallSpacing() * static_cast<float>(c.wall_count),
                    c.wall_recycle_z - c.wall_spawn_z);
}

} // namespace
