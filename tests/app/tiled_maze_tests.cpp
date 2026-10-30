#include <gtest/gtest.h>

import std;

import MazeGenerator.TiledMaze;

namespace {

using MazeGenerator::TiledMaze;

TEST(TiledMazeTest, CloseNorthAndWestSetTheirOwnBits) {
    TiledMaze maze = TiledMaze::Create(4, 4);
    maze.OpenNorth(1, 2);
    maze.OpenWest(1, 2);
    // Leave neighbouring walls open so a write to the wrong bit shows up.
    maze.OpenNorth(1, 1);
    maze.OpenWest(0, 2);

    ASSERT_FALSE(maze.North(1, 2));
    ASSERT_FALSE(maze.West(1, 2));

    maze.CloseNorth(1, 2);
    maze.CloseWest(1, 2);

    EXPECT_TRUE(maze.North(1, 2));
    EXPECT_TRUE(maze.West(1, 2));
    EXPECT_FALSE(maze.North(1, 1));
    EXPECT_FALSE(maze.West(0, 2));
}

TEST(TiledMazeTest, CloseIsIdempotent) {
    TiledMaze maze = TiledMaze::Create(4, 4);
    const TiledMaze closed = maze;

    maze.CloseNorth(0, 0);
    maze.CloseWest(0, 0);

    EXPECT_EQ(maze, closed);
}

TEST(TiledMazeTest, CloseSouthAndEastRedirectToTheNeighbour) {
    TiledMaze maze = TiledMaze::Create(4, 4);
    maze.OpenSouth(3, 3); // Hits the extra boundary row.
    maze.OpenEast(3, 2);  // Hits the extra boundary column.

    ASSERT_FALSE(maze.South(3, 3));
    ASSERT_FALSE(maze.East(3, 2));

    maze.CloseSouth(3, 3);
    maze.CloseEast(3, 2);

    EXPECT_TRUE(maze.South(3, 3));
    EXPECT_TRUE(maze.East(3, 2));
    // South/east are the north/west walls of the cell below/right.
    EXPECT_TRUE(maze.North(3, 4));
    EXPECT_TRUE(maze.West(4, 2));
}

TEST(TiledMazeTest, OpenThenCloseRestoresTheMazeAcrossTileBoundaries) {
    constexpr std::uint32_t kSize = 64;
    const TiledMaze fresh = TiledMaze::Create(kSize, kSize);
    TiledMaze maze = fresh;

    // Tile corners and adjacent cells, plus the far boundary cell.
    const std::array<std::array<std::uint32_t, 2>, 8> cells{{
        {0, 0},
        {15, 15},
        {16, 15},
        {15, 16},
        {16, 16},
        {31, 31},
        {32, 32},
        {kSize - 1, kSize - 1},
    }};

    for (const auto& cell : cells) {
        const std::uint32_t x = cell[0];
        const std::uint32_t y = cell[1];
        maze.OpenNorth(x, y);
        maze.OpenWest(x, y);
        maze.OpenSouth(x, y);
        maze.OpenEast(x, y);
        EXPECT_FALSE(maze.North(x, y)) << x << "," << y;
        EXPECT_FALSE(maze.West(x, y)) << x << "," << y;
        EXPECT_FALSE(maze.South(x, y)) << x << "," << y;
        EXPECT_FALSE(maze.East(x, y)) << x << "," << y;
    }

    for (const auto& cell : cells) {
        maze.CloseNorth(cell[0], cell[1]);
        maze.CloseWest(cell[0], cell[1]);
        maze.CloseSouth(cell[0], cell[1]);
        maze.CloseEast(cell[0], cell[1]);
    }

    EXPECT_EQ(maze, fresh);
}

TEST(TiledMazeTest, EmptyMazeOpensInteriorAndKeepsBoundaryClosed) {
    constexpr std::uint32_t kSize = 5;
    const TiledMaze maze = TiledMaze::Create(kSize, kSize, false);

    for (std::uint32_t y = 0; y < kSize; ++y) {
        for (std::uint32_t x = 0; x < kSize; ++x) {
            // The outer ring stays closed; every interior passage is open.
            EXPECT_EQ(maze.North(x, y), y == 0) << x << "," << y;
            EXPECT_EQ(maze.South(x, y), y == kSize - 1) << x << "," << y;
            EXPECT_EQ(maze.West(x, y), x == 0) << x << "," << y;
            EXPECT_EQ(maze.East(x, y), x == kSize - 1) << x << "," << y;
        }
    }
}

TEST(TiledMazeTest, FilledCreateMatchesTheDefault) {
    EXPECT_EQ(TiledMaze::Create(5, 5, true), TiledMaze::Create(5, 5));
}

TEST(TiledMazeTest, EmptyMazeKeepsTheBoundaryClosedAtOneCell) {
    const TiledMaze maze = TiledMaze::Create(1, 1, false);
    EXPECT_TRUE(maze.North(0, 0));
    EXPECT_TRUE(maze.West(0, 0));
    EXPECT_TRUE(maze.South(0, 0));
    EXPECT_TRUE(maze.East(0, 0));
}

} // namespace
