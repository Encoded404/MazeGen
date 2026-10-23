#include <gtest/gtest.h>

#include <iterator>
#include <limits>

import MazeGenerator.Solver;
import MazeGenerator.Solvers;
import MazeGenerator.Solvers.AStar;
import MazeGenerator.TiledMaze;

namespace {

using MazeGenerator::AStarSolver;
using MazeGenerator::CellRef;
using MazeGenerator::CreateSolver;
using MazeGenerator::SolverStatus;
using MazeGenerator::TiledMaze;

// Opens every interior wall so the only obstacle is the outer boundary.
void OpenAllInterior(TiledMaze& maze) {
    for (std::uint32_t y = 0; y < maze.Height(); ++y) {
        for (std::uint32_t x = 1; x < maze.Width(); ++x) {
            maze.OpenWest(x, y);
        }
    }
    for (std::uint32_t y = 1; y < maze.Height(); ++y) {
        for (std::uint32_t x = 0; x < maze.Width(); ++x) {
            maze.OpenNorth(x, y);
        }
    }
}

TEST(AStarTest, FindsTheManhattanPathInAnOpenGrid) {
    TiledMaze maze = TiledMaze::Create(16, 12);
    OpenAllInterior(maze);

    AStarSolver solver;
    [[maybe_unused]] const auto initial = solver.Reset(maze, CellRef{0, 0}, CellRef{15, 11});

    MazeGenerator::SolverStepResult result;
    for (int guard = 0; guard < 1000; ++guard) {
        result = solver.Step(maze, 1000);
        if (result.status != SolverStatus::Running) break;
    }

    ASSERT_EQ(result.status, SolverStatus::Found);
    EXPECT_EQ(result.stats.path_length, 15U + 11U + 1U);
    // The f-tie-break prefers the larger g, so A* walks the path instead
    // of expanding the whole start/goal rectangle (192 cells here).
    EXPECT_LE(result.stats.expanded, 32U);
}

TEST(AStarTest, StepHonoursTheExpansionBudget) {
    TiledMaze maze = TiledMaze::Create(32, 32);
    OpenAllInterior(maze);

    AStarSolver solver;
    [[maybe_unused]] const auto initial = solver.Reset(maze, CellRef{0, 0}, CellRef{31, 31});
    const auto result = solver.Step(maze, 1);

    EXPECT_EQ(result.status, SolverStatus::Running);
    EXPECT_EQ(result.stats.expanded, 1U);
    EXPECT_FALSE(result.changes.empty());
}

TEST(AStarTest, UnlimitedBudgetFinishesInOneStep) {
    TiledMaze maze = TiledMaze::Create(64, 64);
    OpenAllInterior(maze);

    AStarSolver solver;
    [[maybe_unused]] const auto initial = solver.Reset(maze, CellRef{0, 0}, CellRef{63, 63});

    // The app hands the slider's far-right "Unlimited" position to Step() as
    // UINT32_MAX; one call must then run to completion rather than return
    // Running. The budget only has to exceed the cell count, which holds for
    // even the largest allowed maze (65535^2 cells).
    const auto result =
        solver.Step(maze, std::numeric_limits<std::uint32_t>::max());

    EXPECT_EQ(result.status, SolverStatus::Found);
    EXPECT_GT(result.stats.path_length, 1U);
}

TEST(AStarTest, StartEqualsGoalFinishesImmediately) {
    const TiledMaze maze = TiledMaze::Create(8, 8);

    AStarSolver solver;
    [[maybe_unused]] const auto initial = solver.Reset(maze, CellRef{3, 3}, CellRef{3, 3});
    const auto result = solver.Step(maze, 10);

    EXPECT_EQ(result.status, SolverStatus::Found);
    EXPECT_EQ(result.stats.path_length, 1U);
    EXPECT_EQ(result.stats.expanded, 1U);
}

TEST(AStarTest, ReportsExhaustedWhenTheGoalIsUnreachable) {
    const TiledMaze maze = TiledMaze::Create(8, 8); // every wall closed

    AStarSolver solver;
    [[maybe_unused]] const auto initial = solver.Reset(maze, CellRef{0, 0}, CellRef{7, 7});
    const auto result = solver.Step(maze, 100);

    EXPECT_EQ(result.status, SolverStatus::Exhausted);
    EXPECT_EQ(result.stats.expanded, 1U);
}

TEST(AStarTest, HandlesTheRandomMazesBoundaryOpenings) {
    const TiledMaze maze = TiledMaze::CreateRandom(64, 64, 0.85, 1234);

    CellRef start{0, 0};
    CellRef goal{63, 63};
    for (std::uint32_t x = 0; x < maze.Width(); ++x) {
        if (!maze.North(x, 0)) start = {x, 0};
        if (!maze.South(x, maze.Height() - 1)) goal = {x, maze.Height() - 1};
    }

    AStarSolver solver;
    [[maybe_unused]] const auto initial = solver.Reset(maze, start, goal);

    MazeGenerator::SolverStepResult result;
    for (int guard = 0; guard < 10000; ++guard) {
        result = solver.Step(maze, 100000);
        if (result.status != SolverStatus::Running) break;
    }
    EXPECT_EQ(result.status, SolverStatus::Found);
    EXPECT_GT(result.stats.path_length, 1U);
}

TEST(SolverRegistryTest, ExposesAStar) {
    ASSERT_GE(std::size(MazeGenerator::kSolverTypes), 1U);
    EXPECT_EQ(MazeGenerator::kSolverTypes[0].name, "A*");

    auto solver = CreateSolver(0);
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->Name(), "A*");
    EXPECT_EQ(CreateSolver(999), nullptr);
}

} // namespace
