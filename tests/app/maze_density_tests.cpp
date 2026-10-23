#include <gtest/gtest.h>

import MazeGenerator.MazeDensity;
import MazeGenerator.TiledMaze;

namespace {

using MazeGenerator::BuildMazeDensity;
using MazeGenerator::TiledMaze;

TEST(MazeDensityTest, RawBytesMatchesTileLayout) {
    const TiledMaze maze = TiledMaze::Create(17, 33);
    EXPECT_EQ(maze.TilesX(), 2U);
    EXPECT_EQ(maze.TilesY(), 3U);
    EXPECT_EQ(maze.RawBytes().size(), static_cast<std::size_t>(2 * 3 * 64));
    EXPECT_EQ(maze.RawBytes().size() % 4, 0U);
}

TEST(MazeDensityTest, SingleBlockIncludesAllFourBorders) {
    const TiledMaze maze = TiledMaze::Create(4, 4);
    const auto density = BuildMazeDensity(maze);

    ASSERT_EQ(density.base_width, 1U);
    ASSERT_EQ(density.base_height, 1U);
    ASSERT_EQ(density.mip_count, 1U);
    ASSERT_EQ(density.values.size(), 1U);
    // 16 north + 16 west + 4 south boundary + 4 east boundary, over 2*16 slots.
    EXPECT_FLOAT_EQ(density.values[0], 1.25F);
}

TEST(MazeDensityTest, MipChainAveragesTheEdgeBlocks) {
    const TiledMaze maze = TiledMaze::Create(8, 8);
    const auto density = BuildMazeDensity(maze);

    ASSERT_EQ(density.base_width, 2U);
    ASSERT_EQ(density.base_height, 2U);
    ASSERT_EQ(density.mip_count, 2U);
    ASSERT_EQ(density.values.size(), 5U);
    EXPECT_FLOAT_EQ(density.values[0], 1.0F);
    EXPECT_FLOAT_EQ(density.values[1], 1.125F);
    EXPECT_FLOAT_EQ(density.values[2], 1.125F);
    EXPECT_FLOAT_EQ(density.values[3], 1.25F);
    EXPECT_FLOAT_EQ(density.values[4], 1.125F);
}

TEST(MazeDensityTest, OpeningAWallLowersTheBlockDensity) {
    TiledMaze maze = TiledMaze::Create(4, 4);
    maze.OpenNorth(0, 0);

    const auto density = BuildMazeDensity(maze);
    ASSERT_EQ(density.values.size(), 1U);
    EXPECT_FLOAT_EQ(density.values[0], 39.0F / 32.0F);
}

TEST(MazeDensityTest, RandomMazeIsSeedDeterministic) {
    const auto first = TiledMaze::CreateRandom(64, 64, 0.45, 1234);
    const auto second = TiledMaze::CreateRandom(64, 64, 0.45, 1234);
    const auto different = TiledMaze::CreateRandom(64, 64, 0.45, 5678);

    EXPECT_EQ(first, second);
    EXPECT_NE(first, different);
}

} // namespace
