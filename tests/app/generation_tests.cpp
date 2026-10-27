#include <gtest/gtest.h>

#include <iterator>

import MazeGenerator.Generation;
import MazeGenerator.Generators;
import MazeGenerator.Generators.Random;
import MazeGenerator.TiledMaze;

namespace {

using MazeGenerator::GenerateRandomMaze;
using MazeGenerator::MazeGenSettings;
using MazeGenerator::MazeSpec;
using MazeGenerator::TiledMaze;

TEST(GenerationTest, RandomIsSeedDeterministic) {
    const MazeSpec spec{.width = 64, .height = 64, .seed = 1234};
    const MazeGenSettings settings{};

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
    MazeGenSettings settings{};
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
    MazeGenSettings settings{};
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
        MazeSpec{.width = 0, .height = 8, .seed = 0}, MazeGenSettings{});
    EXPECT_FALSE(maze.has_value());
}

TEST(GeneratorRegistryTest, ExposesRegisteredGenerators) {
    ASSERT_EQ(std::size(MazeGenerator::kMazeGenerators), 2U);
    EXPECT_EQ(MazeGenerator::kMazeGenerators[0].name, "Random");
    ASSERT_EQ(MazeGenerator::kMazeGenerators[0].floats.size(), 1U);
    EXPECT_NE(MazeGenerator::kMazeGenerators[0].generate, nullptr);
    EXPECT_EQ(MazeGenerator::kMazeGenerators[1].name, "CustomGeneratorV1");
    EXPECT_NE(MazeGenerator::kMazeGenerators[1].generate, nullptr);
}

} // namespace
