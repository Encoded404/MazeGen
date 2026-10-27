#include <gtest/gtest.h>

import std;

import MazeGenerator.PackedCellGrid;

namespace {

using MazeGenerator::PackedCellGrid;
using MazeGenerator::SolverStateGrid;
using MazeGenerator::VisitedGrid;

TEST(PackedCellGridTest, IsZeroInitializedAndReadsAcrossAWordBoundary) {
    VisitedGrid grid = VisitedGrid::Create(70, 3);

    ASSERT_EQ(grid.Width(), 70U);
    ASSERT_EQ(grid.Height(), 3U);
    ASSERT_EQ(grid.CellCount(), 210U);

    EXPECT_EQ(grid.Get(0, 0), 0U);
    EXPECT_EQ(grid.Get(63, 0), 0U);
    EXPECT_EQ(grid.Get(64, 0), 0U);

    // 64 cells per word, so x = 63 and x = 64 are adjacent cells in different
    // words of the same row.
    grid.Set(63, 0, 1);
    grid.Set(64, 0, 1);
    grid.Set(2, 2, 1);

    EXPECT_EQ(grid.Get(63, 0), 1U);
    EXPECT_EQ(grid.Get(64, 0), 1U);
    EXPECT_EQ(grid.Get(2, 2), 1U);
    EXPECT_EQ(grid.Get(62, 0), 0U);
    EXPECT_EQ(grid.Get(65, 0), 0U);
}

TEST(PackedCellGridTest, SettingABitDoesNotClobberItsNeighbours) {
    VisitedGrid grid = VisitedGrid::Create(8, 1);

    grid.Set(0, 0, 1);
    grid.Set(1, 0, 1);
    grid.Set(2, 0, 1);
    grid.Set(1, 0, 0);

    EXPECT_EQ(grid.Get(0, 0), 1U);
    EXPECT_EQ(grid.Get(1, 0), 0U);
    EXPECT_EQ(grid.Get(2, 0), 1U);
    EXPECT_FALSE(grid.Test(1, 0));
    EXPECT_TRUE(grid.Test(2, 0));
}

TEST(PackedCellGridTest, TwoBitValuesSurviveAdjacentWrites) {
    // 32 cells per word for 2-bit cells, so x = 31 and x = 32 are in different
    // words of the same row.
    SolverStateGrid grid = SolverStateGrid::Create(34, 2);

    grid.Set(0, 0, 2);
    grid.Set(31, 0, 1);
    grid.Set(32, 0, 3);
    grid.Set(33, 1, 2);

    EXPECT_EQ(grid.Get(0, 0), 2U);
    EXPECT_EQ(grid.Get(31, 0), 1U);
    EXPECT_EQ(grid.Get(32, 0), 3U);
    EXPECT_EQ(grid.Get(33, 1), 2U);

    // Overwriting one field must leave the other fields in both words alone.
    grid.Set(31, 0, 0);
    EXPECT_EQ(grid.Get(31, 0), 0U);
    EXPECT_EQ(grid.Get(32, 0), 3U);
    EXPECT_EQ(grid.Get(0, 0), 2U);
}

TEST(PackedCellGridTest, FourBitValuesSpanTheWholeField) {
    PackedCellGrid<4> grid = PackedCellGrid<4>::Create(18, 1);

    grid.Set(0, 0, 0x0U);
    grid.Set(15, 0, 0xFU);
    grid.Set(16, 0, 0x5U);
    grid.Set(17, 0, 0xAU);

    EXPECT_EQ(grid.Get(0, 0), 0x0U);
    EXPECT_EQ(grid.Get(15, 0), 0xFU);
    EXPECT_EQ(grid.Get(16, 0), 0x5U);
    EXPECT_EQ(grid.Get(17, 0), 0xAU);

    grid.Set(16, 0, 0x3U);
    EXPECT_EQ(grid.Get(15, 0), 0xFU);
    EXPECT_EQ(grid.Get(16, 0), 0x3U);
    EXPECT_EQ(grid.Get(17, 0), 0xAU);
}

TEST(PackedCellGridTest, ClearAndResetZeroEveryCell) {
    VisitedGrid grid = VisitedGrid::Create(8, 8);
    grid.Set(3, 3, 1);
    grid.Set(7, 7, 1);

    grid.Clear();
    EXPECT_FALSE(grid.Test(3, 3));
    EXPECT_FALSE(grid.Test(7, 7));

    grid.Set(3, 3, 1);
    grid.Reset(4, 4);

    EXPECT_EQ(grid.Width(), 4U);
    EXPECT_EQ(grid.Height(), 4U);
    EXPECT_EQ(grid.CellCount(), 16U);
    EXPECT_FALSE(grid.Test(3, 3));
    EXPECT_EQ(grid.Get(0, 0), 0U);
}

TEST(PackedCellGridTest, EqualityComparesDimensionsAndContents) {
    VisitedGrid first = VisitedGrid::Create(8, 8);
    VisitedGrid second = VisitedGrid::Create(8, 8);
    EXPECT_EQ(first, second);

    second.Set(5, 5, 1);
    EXPECT_NE(first, second);

    first.Set(5, 5, 1);
    EXPECT_EQ(first, second);

    // Same cell count but different dimensions must not compare equal.
    const VisitedGrid wide = VisitedGrid::Create(16, 4);
    const VisitedGrid tall = VisitedGrid::Create(4, 16);
    EXPECT_EQ(wide.CellCount(), tall.CellCount());
    EXPECT_NE(wide, tall);
}

} // namespace
