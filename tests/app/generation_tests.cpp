#include <gtest/gtest.h>

#include <any>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
using MazeGenerator::GenerateCustomGeneratorV1;
using MazeGenerator::GenerateRandomMaze;
using MazeGenerator::kMazeGenerators;
using MazeGenerator::MazeGeneratorType;
using MazeGenerator::MazeSpec;
using MazeGenerator::RandomSettings;
using MazeGenerator::SettingsVisitor;
using MazeGenerator::TiledMaze;
using MazeGenerator::VisitCustomGeneratorV1Settings;
using MazeGenerator::VisitRandomSettings;

// Returns the column of the single open wall along one row boundary, or
// `width` when that boundary has no opening.
std::uint32_t BoundaryOpeningX(const TiledMaze& maze, bool top) {
    for (std::uint32_t x = 0; x < maze.Width(); ++x) {
        const bool open = top ? !maze.North(x, 0)
                              : !maze.South(x, maze.Height() - 1);
        if (open) {
            return x;
        }
    }
    return maze.Width();
}

// Records the schema a generator declares without rendering anything. This is
// the non-UI visitor the registry tests use; the app supplies the ImGui one.
class RecordingVisitor final : public SettingsVisitor {
public:
    struct Field {
        std::string label;
        std::string kind;
        FieldHint hint = FieldHint::Slider;
        float min = 0.0f;
        float max = 0.0f;
        std::string format;
    };

    std::vector<Field> fields;
    std::vector<std::string> groups;
    std::vector<std::string> helps;

    void Float(const char* label, float&, float min, float max,
               const char* format, FieldHint hint) override {
        fields.push_back({label, "float", hint, min, max, format});
    }
    void Int(const char* label, int&, int, int, FieldHint hint) override {
        fields.push_back({label, "int", hint});
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

// True when every in-grid cell can be reached from the top entrance through
// open walls. The generator only ever jumps to a cell that has an open wall
// back into the visited set, so "visited" and "reachable" agree.
bool AllCellsReachableFromTop(const TiledMaze& maze) {
    const std::uint32_t width = maze.Width();
    const std::uint32_t height = maze.Height();
    const std::uint32_t top_x = BoundaryOpeningX(maze, true);
    if (top_x >= width) {
        return false;
    }

    const auto index = [width](std::uint32_t x, std::uint32_t y) {
        return static_cast<std::size_t>(y) * width + x;
    };

    std::vector<bool> seen(static_cast<std::size_t>(width) * height, false);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> stack;
    seen[index(top_x, 0)] = true;
    stack.emplace_back(top_x, 0);

    while (!stack.empty()) {
        const auto [x, y] = stack.back();
        stack.pop_back();

        const auto visit = [&](std::uint32_t nx, std::uint32_t ny) {
            if (seen[index(nx, ny)]) {
                return;
            }
            seen[index(nx, ny)] = true;
            stack.emplace_back(nx, ny);
        };
        if (x > 0 && !maze.West(x, y)) {
            visit(x - 1, y);
        }
        if (x + 1 < width && !maze.East(x, y)) {
            visit(x + 1, y);
        }
        if (y > 0 && !maze.North(x, y)) {
            visit(x, y - 1);
        }
        if (y + 1 < height && !maze.South(x, y)) {
            visit(x, y + 1);
        }
    }

    for (const bool reached : seen) {
        if (!reached) {
            return false;
        }
    }
    return true;
}

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

    ASSERT_EQ(visitor.fields.size(), 4U);
    EXPECT_EQ(visitor.fields[0].label, "Stop chance scaler");
    EXPECT_EQ(visitor.fields[0].kind, "float");
    EXPECT_EQ(visitor.fields[1].label, "Branch chance");
    EXPECT_EQ(visitor.fields[1].kind, "float");
    EXPECT_EQ(visitor.fields[2].label, "Hole Fill Threshold");
    EXPECT_EQ(visitor.fields[2].kind, "float");
    EXPECT_EQ(visitor.fields[3].label, "Hole Open Fraction");
    EXPECT_EQ(visitor.fields[3].kind, "float");
    EXPECT_TRUE(visitor.groups.empty());
    EXPECT_FLOAT_EQ(settings.stop_chance_scaler, 1.0f);
    EXPECT_FLOAT_EQ(settings.branch_chance, 0.2f);
    EXPECT_FLOAT_EQ(settings.hole_fill_threshold, 0.2f);
    EXPECT_FLOAT_EQ(settings.hole_open_fraction, 0.075f);
}

TEST(GeneratorRegistryTest, HoleFillThresholdUsesALogarithmicSlider) {
    // The threshold is picked from a narrow band of small fractions, so it is
    // the one setting that opts into a logarithmic track. ImGui derives the
    // track's non-zero floor from the format's decimal count (10^-digits), so
    // the format is part of the contract: a coarser one would strand the lower
    // end of the range at the left edge and still print it as zero. The minimum
    // stays 0.0 so "reopen every hole" is still drag-reachable at the far left.
    CustomGeneratorV1Settings settings{};
    RecordingVisitor visitor;
    VisitCustomGeneratorV1Settings(settings, visitor);

    ASSERT_EQ(visitor.fields.size(), 4U);
    const RecordingVisitor::Field& threshold = visitor.fields[2];
    EXPECT_EQ(threshold.hint, FieldHint::LogSlider);
    EXPECT_EQ(threshold.format, "%.4f");
    EXPECT_FLOAT_EQ(threshold.min, 0.0f);
    EXPECT_FLOAT_EQ(threshold.max, 1.0f);

    // The other fields stay linear.
    EXPECT_EQ(visitor.fields[0].hint, FieldHint::Slider);
    EXPECT_EQ(visitor.fields[1].hint, FieldHint::Slider);
    EXPECT_EQ(visitor.fields[3].hint, FieldHint::Slider);
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

TEST(GenerationTest, CustomGeneratorIsSeedDeterministic) {
    const MazeSpec spec{.width = 64, .height = 64, .seed = 1234};
    const CustomGeneratorV1Settings settings{};

    const auto first = GenerateCustomGeneratorV1(spec, settings);
    const auto second = GenerateCustomGeneratorV1(spec, settings);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*first, *second);

    bool any_different = false;
    for (const std::uint32_t seed : {0U, 1U, 2U, 3U}) {
        const auto other = GenerateCustomGeneratorV1(
            MazeSpec{.width = 64, .height = 64, .seed = seed}, settings);
        ASSERT_TRUE(other.has_value());
        any_different = any_different || (*other != *first);
    }
    EXPECT_TRUE(any_different);
}

TEST(GenerationTest, CustomGeneratorKeepsBoundaryClosedExceptEntrances) {
    // A zero threshold forces the hole pass to run, which is exactly when a bug
    // could open a wall to the outside.
    const auto maze = GenerateCustomGeneratorV1(
        MazeSpec{.width = 64, .height = 64, .seed = 1234},
        CustomGeneratorV1Settings{.hole_fill_threshold = 0.0f});
    ASSERT_TRUE(maze.has_value());

    const std::uint32_t top_x = BoundaryOpeningX(*maze, true);
    const std::uint32_t bottom_x = BoundaryOpeningX(*maze, false);
    ASSERT_LT(top_x, maze->Width());
    ASSERT_LT(bottom_x, maze->Width());
    EXPECT_EQ(top_x + bottom_x, maze->Width() - 1);

    for (std::uint32_t x = 0; x < maze->Width(); ++x) {
        EXPECT_EQ(maze->North(x, 0), x != top_x) << x;
        EXPECT_EQ(maze->South(x, maze->Height() - 1), x != bottom_x) << x;
    }
    for (std::uint32_t y = 0; y < maze->Height(); ++y) {
        EXPECT_TRUE(maze->West(0, y)) << y;
        EXPECT_TRUE(maze->East(maze->Width() - 1, y)) << y;
    }
}

TEST(GenerationTest, CustomGeneratorReopensLargeHoles) {
    // A zero threshold means every unvisited cell is a hole worth reopening, so
    // a correct run leaves nothing unvisited. A zero open fraction also checks
    // that a sealed hole still gets its one required door.
    const MazeSpec spec{.width = 32, .height = 32, .seed = 1234};
    const auto unopened = GenerateCustomGeneratorV1(
        spec, CustomGeneratorV1Settings{.hole_fill_threshold = 1.0f});
    const auto reopened = GenerateCustomGeneratorV1(
        spec, CustomGeneratorV1Settings{.hole_fill_threshold = 0.0f,
                                        .hole_open_fraction = 0.0f});
    ASSERT_TRUE(unopened.has_value());
    ASSERT_TRUE(reopened.has_value());

    // Without the hole pass the walk leaves sealed pockets behind, which is
    // exactly what the pass exists to fix.
    EXPECT_FALSE(AllCellsReachableFromTop(*unopened));
    EXPECT_TRUE(AllCellsReachableFromTop(*reopened));
}

TEST(GenerationTest, HoleFillThresholdAboveOneLeavesGenerationAlone) {
    // Nothing can hold more than the whole grid, so the hole pass must not run
    // and the open fraction must have no effect on the result.
    const MazeSpec spec{.width = 32, .height = 32, .seed = 99};
    const auto no_doors = GenerateCustomGeneratorV1(
        spec, CustomGeneratorV1Settings{.hole_fill_threshold = 1.0f,
                                        .hole_open_fraction = 0.0f});
    const auto all_doors = GenerateCustomGeneratorV1(
        spec, CustomGeneratorV1Settings{.hole_fill_threshold = 1.0f,
                                        .hole_open_fraction = 1.0f});
    ASSERT_TRUE(no_doors.has_value());
    ASSERT_TRUE(all_doors.has_value());
    EXPECT_EQ(*no_doors, *all_doors);
}

TEST(GenerationTest, CustomGeneratorHandlesEdgeDimensionsAndExtremes) {
    const std::pair<std::uint32_t, std::uint32_t> sizes[] = {
        {1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 7}, {8, 8}};
    const CustomGeneratorV1Settings modes[] = {
        CustomGeneratorV1Settings{.stop_chance_scaler = 0.0f, .branch_chance = 0.0f},
        CustomGeneratorV1Settings{.stop_chance_scaler = 2.0f, .branch_chance = 1.0f}};

    for (const auto& [width, height] : sizes) {
        for (const std::uint32_t seed : {0U, 1U, 2U}) {
            for (const CustomGeneratorV1Settings& settings : modes) {
                const auto maze = GenerateCustomGeneratorV1(
                    MazeSpec{.width = width, .height = height, .seed = seed},
                    settings);
                ASSERT_TRUE(maze.has_value())
                    << width << "x" << height << " seed " << seed;
                EXPECT_EQ(maze->Width(), width);
                EXPECT_EQ(maze->Height(), height);
            }
        }
    }
}

TEST(GenerationTest, CustomGeneratorBranchChanceChangesTheWalk) {
    const MazeSpec spec{.width = 64, .height = 64, .seed = 1234};
    const auto no_branches = GenerateCustomGeneratorV1(
        spec, CustomGeneratorV1Settings{.stop_chance_scaler = 1.0f,
                                        .branch_chance = 0.0f});
    const auto all_branches = GenerateCustomGeneratorV1(
        spec, CustomGeneratorV1Settings{.stop_chance_scaler = 1.0f,
                                        .branch_chance = 1.0f});
    ASSERT_TRUE(no_branches.has_value());
    ASSERT_TRUE(all_branches.has_value());
    EXPECT_NE(*no_branches, *all_branches);
}

} // namespace
