// Common maze-solver contract.
//
// A solver is an incremental state machine: Reset() builds its search state and
// returns the initial visualization changes, Step() advances at most
// `max_expansions` nodes and returns the changes since the previous step.
//
// The change stream is deliberately visualization-agnostic. The solver never
// sees the debug grid, the renderer, or ImGui; a consumer applies the flags to
// whatever state it owns. One expansion is one node removed from the frontier,
// which is the common pacing unit for A*, BFS, Dijkstra, DFS, and friends.

module;

export module MazeGenerator.Solver;

import std;

import MazeGenerator.TiledMaze;

export namespace MazeGenerator {

using CellIndex = std::uint32_t;

struct CellRef {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
};

// The four states a search cell can be in. The numeric values are also the bit
// layout DebugGrid stores, so the change stream maps onto the debug byte
// without a translation table.
enum class SolverCellFlag : std::uint8_t {
    None    = 0,
    Open    = 1u << 0, // in the frontier / priority queue
    Closed  = 1u << 1, // expanded
    Path    = 1u << 2, // on the final path
    Current = 1u << 3, // popped on this step
};

[[nodiscard]] constexpr SolverCellFlag operator|(SolverCellFlag a,
                                                 SolverCellFlag b) noexcept {
    return static_cast<SolverCellFlag>(static_cast<std::uint8_t>(a) |
                                       static_cast<std::uint8_t>(b));
}

[[nodiscard]] constexpr bool HasFlag(SolverCellFlag value,
                                     SolverCellFlag bit) noexcept {
    return (static_cast<std::uint8_t>(value) &
            static_cast<std::uint8_t>(bit)) != 0;
}

// parent_dir sentinels: a change either sets an arrow, clears it, or leaves it
// alone. The path pass must leave existing arrows alone.
inline constexpr std::int8_t kNoParentDir = -1;
inline constexpr std::int8_t kKeepParentDir = -2;

// One cell's new visualization state. `flags` is the full resulting state of
// the low flag bits, not a delta.
struct SolverCellChange {
    CellIndex index = 0;
    SolverCellFlag flags = SolverCellFlag::None;
    std::int8_t parent_dir = kKeepParentDir; // 0=N 1=E 2=S 3=W, parent -> cell
};

enum class SolverStatus : std::uint8_t {
    Running,
    Found,
    Exhausted, // the frontier drained without reaching the goal
};

struct SolverStats {
    std::uint64_t expanded = 0;
    std::uint64_t frontier = 0;
    std::uint64_t path_length = 0;
};

struct SolverStepResult {
    SolverStatus status = SolverStatus::Running;
    // Valid until the solver's next Reset/Step.
    std::span<const SolverCellChange> changes{};
    SolverStats stats{};
};

class IMazeSolver {
public:
    virtual ~IMazeSolver() = default;

    [[nodiscard]] virtual std::string_view Name() const = 0;

    // Rebuilds the search for `maze`. Returns the initial changes (the start
    // cell entering the frontier) so a visualization starts in sync.
    [[nodiscard]] virtual std::span<const SolverCellChange>
    Reset(const TiledMaze& maze, CellRef start, CellRef goal) = 0;

    // Advances at most `max_expansions` nodes. The returned span is valid until
    // the next Reset/Step; a caller that needs to keep it must copy.
    [[nodiscard]] virtual SolverStepResult
    Step(const TiledMaze& maze, std::uint32_t max_expansions) = 0;
};

} // namespace MazeGenerator
