// A* over the maze's cell grid: four-neighbour movement, unit edge cost,
// Manhattan heuristic, and a deterministic tie-break so a seeded maze always
// produces the same expansion order.
//
// The visualization bookkeeping is entirely in Emit(): the algorithm itself
// only knows g/parent/open/closed.

module;

export module MazeGenerator.Solvers.AStar;

import std;

import MazeGenerator.Solver;
import MazeGenerator.TiledMaze;

export namespace MazeGenerator {

class AStarSolver final : public IMazeSolver {
public:
    [[nodiscard]] std::string_view Name() const override { return "A*"; }

    [[nodiscard]] std::span<const SolverCellChange>
    Reset(const TiledMaze& maze, CellRef start, CellRef goal) override;

    [[nodiscard]] SolverStepResult
    Step(const TiledMaze& maze, std::uint32_t max_expansions) override;

private:
    struct OpenNode {
        std::uint32_t f = 0;
        std::uint32_t g = 0;
        CellIndex cell = 0;
    };
    struct Greater {
        // Smallest f first. On an f tie, prefer the larger g (closer to the
        // goal): in an open grid every cell of the start/goal rectangle shares
        // the same f, and without this tie-break A* expands the whole
        // rectangle instead of walking the path.
        bool operator()(const OpenNode& a, const OpenNode& b) const noexcept {
            if (a.f != b.f) return a.f > b.f;
            if (a.g != b.g) return a.g < b.g;
            return a.cell > b.cell;
        }
    };
    using OpenQueue = std::priority_queue<OpenNode, std::vector<OpenNode>, Greater>;

    static constexpr CellIndex kInvalidCell =
        std::numeric_limits<CellIndex>::max();
    static constexpr std::uint32_t kUnreachable =
        std::numeric_limits<std::uint32_t>::max();
    static constexpr std::uint8_t kUnseen = 0;
    static constexpr std::uint8_t kOpen = 1;
    static constexpr std::uint8_t kClosed = 2;

    [[nodiscard]] std::uint32_t Heuristic(CellIndex cell) const noexcept;
    void Relax(CellIndex from, CellIndex to, std::int8_t dir);
    void Emit(CellIndex cell, SolverCellFlag flags,
              std::int8_t parent_dir = kKeepParentDir);
    void BuildPath();

    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    CellIndex start_ = 0;
    CellIndex goal_ = 0;
    CellIndex current_ = kInvalidCell;

    std::vector<std::uint32_t> g_;
    std::vector<std::int32_t> parent_;
    std::vector<std::uint8_t> state_; // kUnseen / kOpen / kClosed
    OpenQueue open_;

    std::vector<SolverCellChange> changes_;
    SolverStats stats_{};
    bool found_ = false;
};

inline std::span<const SolverCellChange>
AStarSolver::Reset(const TiledMaze& maze, CellRef start, CellRef goal) {
    width_ = maze.Width();
    height_ = maze.Height();
    start_ = start.y * width_ + start.x;
    goal_ = goal.y * width_ + goal.x;

    const std::size_t cell_count = static_cast<std::size_t>(width_) * height_;
    g_.assign(cell_count, kUnreachable);
    parent_.assign(cell_count, -1);
    state_.assign(cell_count, kUnseen);
    open_ = OpenQueue{};
    changes_.clear();
    stats_ = {};
    found_ = false;
    current_ = kInvalidCell;

    g_[start_] = 0;
    state_[start_] = kOpen;
    open_.push({Heuristic(start_), 0, start_});
    stats_.frontier = 1;
    Emit(start_, SolverCellFlag::Open);
    return changes_;
}

inline SolverStepResult
AStarSolver::Step(const TiledMaze& maze, std::uint32_t max_expansions) {
    changes_.clear();
    std::uint32_t expanded_now = 0;

    while (expanded_now < max_expansions && !open_.empty() && !found_) {
        const OpenNode node = open_.top();
        open_.pop();

        const CellIndex cell = node.cell;
        if (state_[cell] == kClosed) {
            continue; // stale queue entry from an earlier, worse g
        }

        state_[cell] = kClosed;
        ++stats_.expanded;
        if (stats_.frontier > 0) {
            --stats_.frontier;
        }
        ++expanded_now;

        if (current_ != kInvalidCell) {
            Emit(current_, SolverCellFlag::Closed); // clear the previous Current
        }
        current_ = cell;
        Emit(cell, SolverCellFlag::Closed | SolverCellFlag::Current);

        if (cell == goal_) {
            found_ = true;
            Emit(current_, SolverCellFlag::Closed);
            current_ = kInvalidCell;
            BuildPath();
            break;
        }

        const std::uint32_t x = cell % width_;
        const std::uint32_t y = cell / width_;
        // The maze's boundary openings are exits, not neighbours: the top row
        // opens North(x, 0) and the bottom row opens South(x, height-1), which
        // would index the extra boundary row/column (or underflow) if followed.
        if (y > 0 && !maze.North(x, y)) Relax(cell, cell - width_, 0);
        if (x + 1 < width_ && !maze.East(x, y)) Relax(cell, cell + 1, 1);
        if (y + 1 < height_ && !maze.South(x, y)) Relax(cell, cell + width_, 2);
        if (x > 0 && !maze.West(x, y)) Relax(cell, cell - 1, 3);
    }

    const SolverStatus status = found_ ? SolverStatus::Found
                                : open_.empty() ? SolverStatus::Exhausted
                                                : SolverStatus::Running;
    return {status, changes_, stats_};
}

inline std::uint32_t AStarSolver::Heuristic(CellIndex cell) const noexcept {
    const std::uint32_t x = cell % width_;
    const std::uint32_t y = cell / width_;
    const std::uint32_t gx = goal_ % width_;
    const std::uint32_t gy = goal_ / width_;
    const std::uint32_t dx = x > gx ? x - gx : gx - x;
    const std::uint32_t dy = y > gy ? y - gy : gy - y;
    return dx + dy;
}

inline void AStarSolver::Relax(CellIndex from, CellIndex to, std::int8_t dir) {
    const std::uint32_t candidate = g_[from] + 1;
    if (candidate >= g_[to]) {
        return;
    }

    g_[to] = candidate;
    parent_[to] = static_cast<std::int32_t>(from);

    if (state_[to] == kUnseen) {
        state_[to] = kOpen;
        ++stats_.frontier;
        Emit(to, SolverCellFlag::Open, dir);
    } else if (state_[to] == kClosed) {
        // Manhattan on unit four-neighbour edges is consistent, so this never
        // happens in practice; keep the state and the visualization coherent
        // if a future variant uses an inconsistent heuristic.
        state_[to] = kOpen;
        ++stats_.frontier;
        Emit(to, SolverCellFlag::Open, dir);
    } else {
        // Already Open: the cell stays orange, but its parent arrow improves.
        Emit(to, SolverCellFlag::Open, dir);
    }
    open_.push({candidate + Heuristic(to), candidate, to});
}

inline void AStarSolver::Emit(CellIndex cell, SolverCellFlag flags,
                              std::int8_t parent_dir) {
    changes_.push_back({cell, flags, parent_dir});
}

inline void AStarSolver::BuildPath() {
    stats_.path_length = 1;
    Emit(goal_, SolverCellFlag::Closed | SolverCellFlag::Path);

    for (std::int32_t cell = parent_[goal_]; cell >= 0; cell = parent_[cell]) {
        Emit(static_cast<CellIndex>(cell),
             SolverCellFlag::Closed | SolverCellFlag::Path);
        ++stats_.path_length;
        if (static_cast<CellIndex>(cell) == start_) {
            break;
        }
    }
}

} // namespace MazeGenerator
