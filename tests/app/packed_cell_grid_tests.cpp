#include <gtest/gtest.h>

import std;

import logiface;
import logiface.ConsoleLogger;

import MazeGenerator.PackedCellGrid;

namespace {

using MazeGenerator::PackedCellGrid;
using MazeGenerator::SolverStateGrid;
using MazeGenerator::VisitedGrid;

// Installs a logger for the duration of a test, restoring the previous one.
class ScopedLogger {
public:
    explicit ScopedLogger(Logiface::Logger& logger)
        : previous_(Logiface::GetLogger()) {
        Logiface::SetLogger(&logger);
    }
    ~ScopedLogger() { Logiface::SetLogger(previous_); }

    ScopedLogger(const ScopedLogger&) = delete;
    ScopedLogger& operator=(const ScopedLogger&) = delete;

private:
    Logiface::Logger* previous_;
};

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

TEST(PackedCellGridTest, RawWordGetterExposesWholeWords) {
    // 210 cells at one bit each: four words, the last one only partly used.
    VisitedGrid grid = VisitedGrid::Create(70, 3);
    ASSERT_EQ(grid.WordCount(), 4U);

    // A freshly created grid is all unvisited, so every raw word is zero.
    for (std::size_t word = 0; word < grid.WordCount(); ++word) {
        EXPECT_EQ(grid.Word(word), 0U) << "word " << word;
    }

    // x = 63 and x = 64 are adjacent cells in different words.
    grid.Set(63, 0, 1);
    EXPECT_EQ(grid.Word(0), std::uint64_t{1} << 63);
    EXPECT_EQ(grid.Word(1), 0U);

    // Linear index 2 * 70 + 2 = 142 lands in word 2 at bit 14.
    grid.Set(2, 2, 1);
    EXPECT_EQ(grid.Word(2), std::uint64_t{1} << 14);

    // Still-untouched words are zero, which is what makes the raw getter a
    // valid "nothing in this word is visited" test.
    EXPECT_EQ(grid.Word(3), 0U);

    // The raw word must agree with the per-cell view, bit for bit.
    for (std::size_t index = 0; index < grid.CellCount(); ++index) {
        const std::size_t word = index / VisitedGrid::kCellsPerWord;
        const std::uint32_t bit =
            static_cast<std::uint32_t>(index % VisitedGrid::kCellsPerWord);
        EXPECT_EQ((grid.Word(word) >> bit) & 1U, grid.Get(index))
            << "cell " << index;
    }
}

TEST(PackedCellGridTest, RawWordGetterWorksForWiderCells) {
    // 32 cells per word at two bits each, so x = 31 and x = 32 straddle words.
    SolverStateGrid grid = SolverStateGrid::Create(34, 2);
    ASSERT_EQ(grid.WordCount(), 3U);

    grid.Set(0, 0, 2);
    EXPECT_EQ(grid.Word(0), 2U);

    grid.Set(31, 0, 3);
    EXPECT_EQ(grid.Word(0), 2U | (std::uint64_t{3} << 62));
    EXPECT_EQ(grid.Word(1), 0U);

    grid.Set(32, 0, 1);
    EXPECT_EQ(grid.Word(1), 1U);

    // A zero word still means every cell in it is in the zero state.
    EXPECT_EQ(grid.Word(2), 0U);
}

TEST(PackedCellGridTest, RawWordPaddingBitsStayZero) {
    VisitedGrid grid = VisitedGrid::Create(70, 3);
    ASSERT_EQ(grid.WordCount(), 4U);

    for (std::uint32_t y = 0; y < grid.Height(); ++y) {
        for (std::uint32_t x = 0; x < grid.Width(); ++x) {
            grid.Set(x, y, 1);
        }
    }

    constexpr std::uint64_t kAllOnes = ~std::uint64_t{0};
    EXPECT_EQ(grid.Word(0), kAllOnes);
    EXPECT_EQ(grid.Word(1), kAllOnes);
    EXPECT_EQ(grid.Word(2), kAllOnes);
    // 210 cells leaves 18 valid bits in the final word. Padding must stay zero:
    // otherwise a zero-word "all unvisited" test could see stale set bits.
    EXPECT_EQ(grid.Word(3), (std::uint64_t{1} << 18) - 1);
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

TEST(PackedCellGridTest, OutOfBoundsAccessReportsTheCoordinates) {
    // The bounds check logs through the project logger before assert fires;
    // the death-test child inherits the installed logger.
    Logiface::ConsoleLogger logger{Logiface::Level::trace};
    const ScopedLogger install_logger{logger};

    VisitedGrid grid = VisitedGrid::Create(4, 3);

    EXPECT_DEATH(static_cast<void>(grid.Get(7, 2)),
                 "cell \\(x=7, y=2\\) is outside the 4x3 grid");
    EXPECT_DEATH(grid.Set(1, 5, 1),
                 "cell \\(x=1, y=5\\) is outside the 4x3 grid");
    EXPECT_DEATH(static_cast<void>(grid.Test(9, 9)),
                 "cell \\(x=9, y=9\\) is outside the 4x3 grid");
}

} // namespace
