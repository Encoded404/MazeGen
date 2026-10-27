// Solver registry. Adding a solver is a new .cppm under app/solvers/
// implementing IMazeSolver plus one row here; the UI combo and the App read
// kSolverTypes.

module;

export module MazeGenerator.Solvers;

import std;

import MazeGenerator.Solver;
import MazeGenerator.Solvers.AStar;

export namespace MazeGenerator {

struct SolverType {
    std::string_view name;
    std::unique_ptr<IMazeSolver> (*create)();
};

inline constexpr SolverType kSolverTypes[] = {
    {"A*", []() -> std::unique_ptr<IMazeSolver> {
         return std::make_unique<AStarSolver>();
     }},
};

[[nodiscard]] inline std::unique_ptr<IMazeSolver> CreateSolver(std::size_t index) {
    if (index >= std::size(kSolverTypes)) {
        return nullptr;
    }
    return kSolverTypes[index].create();
}

} // namespace MazeGenerator
