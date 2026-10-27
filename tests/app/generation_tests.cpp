#include <gtest/gtest.h>

#include <any>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

import MazeGenerator.Generation;
import MazeGenerator.Generators;
import MazeGenerator.Generators.CustomGeneratorV1;
import MazeGenerator.Generators.Random;
import MazeGenerator.TiledMaze;

namespace {

using MazeGenerator::CustomGeneratorV1Settings;
using MazeGenerator::EnumOption;
using MazeGenerator::FieldHint;
using MazeGenerator::GenerateRandomMaze;
using MazeGenerator::kMazeGenerators;
using MazeGenerator::MazeGeneratorType;
using MazeGenerator::MazeSpec;
using MazeGenerator::RandomSettings;
using MazeGenerator::SettingsVisitor;
using MazeGenerator::TiledMaze;
using MazeGenerator::VisitCustomGeneratorV1Settings;
using MazeGenerator::VisitRandomSettings;

// Records the schema a generator declares without rendering anything. This is
// the non-UI visitor the registry tests use; the app supplies the ImGui one.
class RecordingVisitor final : public SettingsVisitor {
public:
    struct Field {
        std::string label;
        std::string kind;
    };

    std::vector<Field> fields;
    std::vector<std::string> groups;
    std::vector<std::string> helps;

    void Float(const char* label, float&, float, float, const char*,
               FieldHint) override {
        fields.push_back({label, "float"});
    }
    void Int(const char* label, int&, int, int, FieldHint) override {
        fields.push_back({label, "int"});
    }
    void Bool(const char* label, bool&) override {
        fields.push_back({label, "bool"});
    }
    void Enum(const char* label, int&, std::span<const EnumOption>) override {
        fields.push_back({label, "enum"});
    }
    void Path(const char* label, std::string&) override {
        fields.push_back({label, "path"});
    }
    bool BeginGroup(const char* label, bool) override {
        groups.push_back(label);
        return true;
    }
    void EndGroup() override {}
    void Help(std::string_view text) override {
        helps.push_back(std::string(text));
    }
};

TEST(GenerationTest, RandomIsSeedDeterministic) {
    const MazeSpec spec{.width = 64, .height = 64, .seed = 1234};
    const RandomSettings settings{};

    const auto first = GenerateRandomMaze(spec, settings);
    const auto second = GenerateRandomMaze(spec, settings);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*first, *second);

    const auto different = GenerateRandomMaze(
        MazeSpec{.width = 64, .height = 64, .seed = 5678}, settings);
    ASSERT_TRUE(different.has_value());
    EXPECT_NE(*first, *different);
}

TEST(GenerationTest, ProbabilityZeroLeavesInteriorClosed) {
    const MazeSpec spec{.width = 8, .height = 8, .seed = 7};
    RandomSettings settings{};
    settings.open_probability = 0.0f;

    const auto maze = GenerateRandomMaze(spec, settings);
    ASSERT_TRUE(maze.has_value());

    // CreateRandom always opens the entrance and exit; every other interior
    // wall stays closed.
    for (std::uint32_t y = 0; y < maze->Height(); ++y) {
        for (std::uint32_t x = 1; x < maze->Width(); ++x) {
            EXPECT_TRUE(maze->West(x, y)) << x << "," << y;
        }
    }
    for (std::uint32_t y = 1; y < maze->Height(); ++y) {
        for (std::uint32_t x = 0; x < maze->Width(); ++x) {
            EXPECT_TRUE(maze->North(x, y)) << x << "," << y;
        }
    }
}

TEST(GenerationTest, ProbabilityOneLeavesInteriorOpen) {
    const MazeSpec spec{.width = 8, .height = 8, .seed = 7};
    RandomSettings settings{};
    settings.open_probability = 1.0f;

    const auto maze = GenerateRandomMaze(spec, settings);
    ASSERT_TRUE(maze.has_value());
    for (std::uint32_t y = 0; y < maze->Height(); ++y) {
        for (std::uint32_t x = 1; x < maze->Width(); ++x) {
            EXPECT_FALSE(maze->West(x, y)) << x << "," << y;
        }
    }
    for (std::uint32_t y = 1; y < maze->Height(); ++y) {
        for (std::uint32_t x = 0; x < maze->Width(); ++x) {
            EXPECT_FALSE(maze->North(x, y)) << x << "," << y;
        }
    }
}

TEST(GenerationTest, RejectsEmptyDimensions) {
    const auto maze = GenerateRandomMaze(
        MazeSpec{.width = 0, .height = 8, .seed = 0}, RandomSettings{});
    EXPECT_FALSE(maze.has_value());
}

TEST(GeneratorRegistryTest, ExposesRegisteredGenerators) {
    ASSERT_EQ(std::size(kMazeGenerators), 2U);
    EXPECT_EQ(kMazeGenerators[0].name, "Random");
    EXPECT_EQ(kMazeGenerators[1].name, "CustomGeneratorV1");
    for (const MazeGeneratorType& generator : kMazeGenerators) {
        EXPECT_NE(generator.generate, nullptr);
        EXPECT_NE(generator.visit, nullptr);
        EXPECT_NE(generator.make_defaults, nullptr);
    }
}

TEST(GeneratorRegistryTest, RandomDescribesItsOwnSettings) {
    RandomSettings settings{};
    RecordingVisitor visitor;
    VisitRandomSettings(settings, visitor);

    ASSERT_EQ(visitor.fields.size(), 1U);
    EXPECT_EQ(visitor.fields[0].label, "Open probability");
    EXPECT_EQ(visitor.fields[0].kind, "float");
    ASSERT_EQ(visitor.helps.size(), 1U);
}

TEST(GeneratorRegistryTest, CustomGeneratorDescribesItsOwnSettings) {
    CustomGeneratorV1Settings settings{};
    RecordingVisitor visitor;
    VisitCustomGeneratorV1Settings(settings, visitor);

    ASSERT_EQ(visitor.fields.size(), 2U);
    EXPECT_EQ(visitor.fields[0].label, "Stop chance scaler");
    EXPECT_EQ(visitor.fields[0].kind, "float");
    EXPECT_EQ(visitor.fields[1].label, "Branch chance");
    EXPECT_EQ(visitor.fields[1].kind, "float");
    EXPECT_TRUE(visitor.groups.empty());
    EXPECT_FLOAT_EQ(settings.stop_chance_scaler, 1.0f);
    EXPECT_FLOAT_EQ(settings.branch_chance, 0.2f);
}

TEST(GeneratorRegistryTest, TypeErasedEntryOwnsItsConcreteSettings) {
    const MazeGeneratorType& random = kMazeGenerators[0];
    std::any settings = random.make_defaults();
    ASSERT_TRUE(settings.has_value());
    EXPECT_FLOAT_EQ(std::any_cast<const RandomSettings&>(settings).open_probability,
                    0.45f);

    RecordingVisitor visitor;
    random.visit(settings, visitor);
    ASSERT_EQ(visitor.fields.size(), 1U);
    EXPECT_EQ(visitor.fields[0].label, "Open probability");

    const auto maze = random.generate(
        MazeSpec{.width = 16, .height = 16, .seed = 9}, settings);
    ASSERT_TRUE(maze.has_value());
    EXPECT_EQ(maze->Width(), 16U);
    EXPECT_EQ(maze->Height(), 16U);
}

} // namespace
