#include <CLI/CLI.hpp>
#include <gtest/gtest.h>

#include <any>
#include <array>
#include <cstddef>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

import MazeGenerator.GeneratorCli;
import MazeGenerator.Generators;
import MazeGenerator.Generators.CustomGeneratorV1;
import MazeGenerator.Generators.Random;
import MazeGenerator.Generation;
import MazeGenerator.Solvers;

namespace {

using MazeGenerator::CliSettingsVisitor;
using MazeGenerator::CustomGeneratorV1Settings;
using MazeGenerator::EnumOption;
using MazeGenerator::GeneratorOptionPrefix;
using MazeGenerator::kMazeGenerators;
using MazeGenerator::RandomSettings;
using MazeGenerator::RegisterGeneratorOptions;
using MazeGenerator::SettingsVisitor;

struct SyntheticSettings {
    float ratio = 0.5f;
    int count = 3;
    bool enabled = false;
    int mode = 1;
    std::string path;
    int nested_count = 0;
};

// Exercises every SettingsVisitor field kind and one nested group.
// No registered generator uses Int/Bool/Enum/Path or groups yet, so this is the coverage that keeps the CLI visitor honest for future generators.
void VisitSynthetic(SyntheticSettings& settings, SettingsVisitor& visitor) {
    visitor.Float({"ratio", "Ratio"}, settings.ratio, 0.0f, 1.0f);
    visitor.Help("How much ratio.");
    visitor.Int({"count", "Count"}, settings.count, 0, 10);
    visitor.Bool({"enabled", "Enabled"}, settings.enabled);

    const std::array<EnumOption, 2> modes{
        EnumOption{.label = "Fast", .value = 1},
        EnumOption{.label = "Slow", .value = 2},
    };
    visitor.Enum({"mode", "Mode"}, settings.mode, modes);
    visitor.Path({"path", "Path"}, settings.path);

    if (visitor.BeginGroup({"advanced", "Advanced"}, true)) {
        visitor.Int({"nested", "Nested"}, settings.nested_count, 0, 10);
        visitor.EndGroup();
    }
}

std::vector<std::any> DefaultSettings() {
    std::vector<std::any> settings;
    settings.reserve(std::size(kMazeGenerators));
    for (const MazeGenerator::MazeGeneratorType& generator : kMazeGenerators) {
        settings.push_back(generator.make_defaults());
    }
    return settings;
}

} // namespace

TEST(GeneratorCliTest, PrefixIsKebabCase) {
    EXPECT_EQ(GeneratorOptionPrefix("Random"), "random");
    EXPECT_EQ(GeneratorOptionPrefix("CustomGeneratorV1"), "custom-generator-v1");
    EXPECT_EQ(GeneratorOptionPrefix("AStar"), "a-star");
    EXPECT_EQ(GeneratorOptionPrefix("IOThing"), "io-thing");
}

TEST(GeneratorCliTest, EveryFieldKindParses) {
    SyntheticSettings settings{};
    CLI::App app{"test"};
    CliSettingsVisitor visitor(app, "test");
    VisitSynthetic(settings, visitor);

    app.parse(
        "--test-ratio 0.25 --test-count 7 --test-enabled --test-mode slow "
        "--test-path /tmp/settings.bin --test-advanced-nested 4");

    EXPECT_FLOAT_EQ(settings.ratio, 0.25f);
    EXPECT_EQ(settings.count, 7);
    EXPECT_TRUE(settings.enabled);
    EXPECT_EQ(settings.mode, 2);
    EXPECT_EQ(settings.path, "/tmp/settings.bin");
    EXPECT_EQ(settings.nested_count, 4);
}

TEST(GeneratorCliTest, AcceptsValuesOutsideTheSuggestedRange) {
    SyntheticSettings settings{};
    CLI::App app{"test"};
    CliSettingsVisitor visitor(app, "test");
    VisitSynthetic(settings, visitor);

    app.parse("--test-ratio 1.5 --test-count 42");

    EXPECT_FLOAT_EQ(settings.ratio, 1.5f);
    EXPECT_EQ(settings.count, 42);
}

TEST(GeneratorCliTest, AttachesHelpTextToTheOption) {
    SyntheticSettings settings{};
    CLI::App app{"test"};
    CliSettingsVisitor visitor(app, "test");
    VisitSynthetic(settings, visitor);

    CLI::Option* option = app.get_option("--test-ratio");
    ASSERT_NE(option, nullptr);
    EXPECT_NE(option->get_description().find("How much ratio."),
              std::string::npos);
    EXPECT_NE(option->get_description().find("Suggested range: 0 to 1."),
              std::string::npos);
}

TEST(GeneratorCliTest, RegistersRegistrySettingsUnderGeneratorPrefixes) {
    std::vector<std::any> settings = DefaultSettings();
    CLI::App app{"test"};
    RegisterGeneratorOptions(app, kMazeGenerators, settings);

    ASSERT_NE(app.get_option("--random-open-probability"), nullptr);
    ASSERT_NE(app.get_option("--custom-generator-v1-stop-chance-scaler"), nullptr);
    ASSERT_NE(app.get_option("--custom-generator-v1-branch-chance"), nullptr);
    ASSERT_NE(app.get_option("--custom-generator-v1-hole-fill-threshold"), nullptr);
    ASSERT_NE(app.get_option("--custom-generator-v1-hole-open-fraction"), nullptr);

    app.parse("--random-open-probability 0.6");
    EXPECT_FLOAT_EQ(
        std::any_cast<RandomSettings&>(settings[0]).open_probability, 0.6f);
    // Untouched settings keep their registry defaults.
    EXPECT_FLOAT_EQ(
        std::any_cast<CustomGeneratorV1Settings&>(settings[1]).branch_chance,
        0.2f);
}

TEST(RegistryLookupTest, FindsRegistryEntriesByName) {
    EXPECT_EQ(MazeGenerator::FindGeneratorIndex("Random"),
              std::optional<std::size_t>{0});
    EXPECT_EQ(MazeGenerator::FindGeneratorIndex("CustomGeneratorV1"),
              std::optional<std::size_t>{1});
    EXPECT_FALSE(MazeGenerator::FindGeneratorIndex("random").has_value());

    EXPECT_EQ(MazeGenerator::FindSolverIndex("A*"),
              std::optional<std::size_t>{0});
    EXPECT_FALSE(MazeGenerator::FindSolverIndex("bfs").has_value());
}
