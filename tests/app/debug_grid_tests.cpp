#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

import MazeGenerator.DebugGrid;
import MazeGenerator.Solver;

namespace {

using MazeGenerator::DebugGrid;
using MazeGenerator::kKeepParentDir;
using MazeGenerator::SolverCellChange;
using MazeGenerator::SolverCellFlag;

TEST(DebugGridTest, ApplyUpdatesCellFlagsAndBlockCounts) {
    DebugGrid grid;
    grid.Reset(16, 16);

    grid.Apply(SolverCellChange{0, SolverCellFlag::Open, kKeepParentDir});
    EXPECT_EQ(grid.Cell(0) & DebugGrid::kFlagMask,
              static_cast<std::uint8_t>(SolverCellFlag::Open));
    EXPECT_EQ(grid.Block(0) & 0xFFu, 1U); // open counter

    grid.Apply(SolverCellChange{0, SolverCellFlag::Closed, kKeepParentDir});
    EXPECT_EQ(grid.Block(0) & 0xFFu, 0U);
    EXPECT_EQ((grid.Block(0) >> 8u) & 0xFFu, 1U); // closed counter

    grid.Apply(SolverCellChange{
        0, SolverCellFlag::Closed | SolverCellFlag::Path, kKeepParentDir});
    EXPECT_EQ((grid.Block(0) >> 16u) & 0xFFu, 1U); // path counter
}

TEST(DebugGridTest, ParentDirectionIsPackedWithoutClobberingFlags) {
    DebugGrid grid;
    grid.Reset(16, 16);

    grid.Apply(SolverCellChange{5, SolverCellFlag::Closed, 2});
    const std::uint8_t cell = grid.Cell(5);
    EXPECT_EQ(cell & DebugGrid::kFlagMask,
              static_cast<std::uint8_t>(SolverCellFlag::Closed));
    EXPECT_EQ(cell & 0x80u, 0x80u);
    EXPECT_EQ((cell >> 5u) & 3u, 2U);

    // A later flag-only change must preserve the arrow.
    grid.Apply(SolverCellChange{5, SolverCellFlag::Closed | SolverCellFlag::Path,
                                kKeepParentDir});
    EXPECT_EQ(grid.Cell(5) & 0x80u, 0x80u);
    EXPECT_EQ((grid.Cell(5) >> 5u) & 3u, 2U);
}

TEST(DebugGridTest, FlushCopiesFullThenIncrementalChanges) {
    DebugGrid grid;
    grid.Reset(16, 16);

    std::vector<std::uint8_t> cells(16 * 16, 0xAA);
    std::vector<std::uint32_t> blocks(4, 0xAA);
    grid.Apply(SolverCellChange{3, SolverCellFlag::Open, kKeepParentDir});
    grid.FlushTo(cells.data(), blocks.data());
    EXPECT_EQ(cells[3], static_cast<std::uint8_t>(SolverCellFlag::Open));
    EXPECT_EQ(cells[0], 0U);
    EXPECT_NE(blocks[0], 0xAAu);

    // Now only the changed cell/block should be written.
    cells[0] = 0xAA;
    grid.Apply(SolverCellChange{10, SolverCellFlag::Closed, kKeepParentDir});
    grid.FlushTo(cells.data(), blocks.data());
    EXPECT_EQ(cells[0], 0xAAu); // untouched by the incremental flush
    EXPECT_EQ(cells[10], static_cast<std::uint8_t>(SolverCellFlag::Closed));
}

TEST(DebugGridTest, ClearResetsCellsAndCounters) {
    DebugGrid grid;
    grid.Reset(16, 16);
    grid.Apply(SolverCellChange{7, SolverCellFlag::Closed, kKeepParentDir});
    grid.Clear();
    EXPECT_EQ(grid.Cell(7), 0U);
    EXPECT_EQ(grid.Block(0), 0U);
}

} // namespace
