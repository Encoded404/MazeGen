// CustomGeneratorV1: a wall-closing random walk over an open maze.
//
// The maze starts with its interior open and its boundary closed. The walk
// enters through the top opening, closes the side walls of each cell it moves
// through, and stops or branches according to the stop and branch chances.
//
// When the walk runs out of branches it leaves unvisited regions behind. Each
// region larger than hole_fill_threshold * width * height is reopened: some of
// its border walls are opened as doors and the walk is seeded inside it, so
// generation resumes instead of ending with a sealed void.

module;

#include <logging/logging_macros.hpp>

export module MazeGenerator.Generators.CustomGeneratorV1;

import std;

import MazeGenerator.Generation;
import MazeGenerator.PackedCellGrid;
import MazeGenerator.TiledMaze;

import logiface;

namespace {
    enum class Direction : std::uint8_t {
        up = 0,
        right = 1,
        down = 2,
        left = 3,
    };

    struct Position {
        std::uint32_t x, y;
    };

    [[nodiscard]] Position MoveInDir(Position pos, Direction dir) {
        Position return_value = pos;
        return_value.x += dir == Direction::left ? -1
                          : dir == Direction::right ? 1
                                                    : 0;
        return_value.y += dir == Direction::up ? -1
                          : dir == Direction::down ? 1
                                                   : 0;
        return return_value;
    }

    [[nodiscard]] bool InBounds(const Position pos, std::uint32_t width,
                                std::uint32_t height) noexcept {
        return pos.x < width && pos.y < height;
    }

    [[nodiscard]] bool GetDirectionState(const Position pos, const Direction dir,
                                         const MazeGenerator::TiledMaze& maze) {
        switch (dir) {
            case Direction::up:
                return maze.North(pos.x, pos.y);
            case Direction::right:
                return maze.East(pos.x, pos.y);
            case Direction::down:
                return maze.South(pos.x, pos.y);
            case Direction::left:
                return maze.West(pos.x, pos.y);
            default:
                return false;
        }
    }

    // True when moving one cell in `dir` stays inside the maze and is not
    // blocked by a wall. The boundary row/column is outside the cell grid, so
    // an open boundary entrance is still not a legal move out of the maze.
    [[nodiscard]] bool CanMove(const Position pos, const Direction dir,
                               const MazeGenerator::TiledMaze& maze,
                               std::uint32_t width, std::uint32_t height) {
        const Position next = MoveInDir(pos, dir);
        return InBounds(next, width, height) &&
               !GetDirectionState(pos, dir, maze);
    }

    // CanMove(), plus "the target cell is still unvisited". The turn logic uses
    // this so a walk resumed inside a reopened hole cannot step back out into
    // already carved maze.
    [[nodiscard]] bool CanMoveIntoUnvisited(
        const Position pos, const Direction dir,
        const MazeGenerator::TiledMaze& maze,
        const MazeGenerator::VisitedGrid& visited, std::uint32_t width,
        std::uint32_t height) {
        if (!CanMove(pos, dir, maze, width, height)) {
            return false;
        }
        const Position next = MoveInDir(pos, dir);
        return !visited.Get(next.x, next.y);
    }

    void CloseDirectionState(const Position pos, const Direction dir,
                             MazeGenerator::TiledMaze& maze) {
        switch (dir) {
            case Direction::up:
                maze.CloseNorth(pos.x, pos.y);
                break;
            case Direction::right:
                maze.CloseEast(pos.x, pos.y);
                break;
            case Direction::down:
                maze.CloseSouth(pos.x, pos.y);
                break;
            case Direction::left:
                maze.CloseWest(pos.x, pos.y);
                break;
        }
    }

    // The mirror of CloseDirectionState. Only called for edges between two
    // in-grid cells, so the boundary row/column redirects never fire.
    void OpenDirectionState(const Position pos, const Direction dir,
                            MazeGenerator::TiledMaze& maze) {
        switch (dir) {
            case Direction::up:
                maze.OpenNorth(pos.x, pos.y);
                break;
            case Direction::right:
                maze.OpenEast(pos.x, pos.y);
                break;
            case Direction::down:
                maze.OpenSouth(pos.x, pos.y);
                break;
            case Direction::left:
                maze.OpenWest(pos.x, pos.y);
                break;
        }
    }

    [[nodiscard]] bool ShouldStop(float stop_chance_scaler,
                                  std::uint32_t move_amount,
                                  std::uint32_t averaged_size,
                                  std::mt19937& rng) {
        const std::uint32_t half_size =
            std::max(averaged_size / 2, std::uint32_t{1});
        const float progress =
            static_cast<float>(move_amount) / static_cast<float>(half_size);
        const float chance =
            std::clamp(progress * stop_chance_scaler, 0.0f, 1.0f);

        std::bernoulli_distribution should_trigger(chance);
        return should_trigger(rng);
    }

    [[nodiscard]] bool ShouldBranch(float branch_chance, std::mt19937& rng) {
        std::bernoulli_distribution should_trigger(
            std::clamp(branch_chance, 0.0f, 1.0f));
        return should_trigger(rng);
    }

    template <std::integral T>
    [[nodiscard]] constexpr T Modulo(T a, T b) noexcept {
        const T result = static_cast<T>(a % b);
        return result < T{0} ? static_cast<T>(result + b) : result;
    }

    [[nodiscard]] constexpr Direction Rotate(Direction dir, std::int8_t amount) noexcept {
        return static_cast<Direction>(Modulo(static_cast<int>(dir) + amount, 4));
    }

    double Distance(const Position a, const Position b) {
        const std::int32_t x_diff = a.x - b.x;
        const std::int32_t y_diff = a.y - b.y;
        return  std::sqrt(std::pow(x_diff, 2) + std::pow(y_diff, 2));
    }

    std::int8_t GetFarthestRotationFromTarget(Position pos, Position target, Direction start_dir) {
        double a_dist = Distance(MoveInDir(pos, Rotate(start_dir, 1)), target);
        double b_dist = Distance(MoveInDir(pos, Rotate(start_dir, -1)), target);
        if (a_dist > b_dist) return 1;
        else return -1;
    }

    // The border of one unvisited hole: every edge between a hole cell and a
    // visited cell. `direction` points from the hole cell toward the visited
    // neighbour, so opening that edge is a way into the hole.
    struct HoleBorder {
        std::vector<std::pair<Position, Direction>> passages; // already open
        std::vector<std::pair<Position, Direction>> walls;    // still closed
    };

    [[nodiscard]] HoleBorder ClassifyHoleBorder(
        std::span<const std::uint32_t> hole,
        const MazeGenerator::VisitedGrid& visited,
        const MazeGenerator::TiledMaze& maze, std::uint32_t width,
        std::uint32_t height) {
        HoleBorder border;
        for (const std::uint32_t index : hole) {
            const Position cell{index % width, index / width};
            for (const Direction dir : {Direction::up, Direction::right,
                                        Direction::down, Direction::left}) {
                const Position next = MoveInDir(cell, dir);
                if (!InBounds(next, width, height) ||
                    !visited.Get(next.x, next.y)) {
                    continue; // outside the maze, or another hole cell
                }
                if (GetDirectionState(cell, dir, maze)) {
                    border.walls.emplace_back(cell, dir);
                } else {
                    border.passages.emplace_back(cell, dir);
                }
            }
        }
        return border;
    }

    // The direction the walk should face after jumping into a hole through the
    // edge (hole_cell, door_dir). It must point at another hole cell: a branch
    // position whose target is visited stops the walk immediately, and the turn
    // logic then has nothing unvisited to turn into. Walls between two
    // unvisited cells are never closed, so any unvisited neighbour is a legal
    // step. A one-cell hole has none; the caller then falls back to the door
    // direction, which still marks the cell visited, so progress is guaranteed.
    [[nodiscard]] std::optional<Direction> InwardDirection(
        const Position hole_cell, const MazeGenerator::VisitedGrid& visited,
        std::uint32_t width, std::uint32_t height, Direction door_dir) {
        for (const Direction dir :
             {Rotate(door_dir, 1), Rotate(door_dir, -1)}) {
            const Position next = MoveInDir(hole_cell, dir);
            if (InBounds(next, width, height) && !visited.Get(next.x, next.y)) {
                return dir;
            }
        }
        return std::nullopt;
    }

    // Calls on_hole once per 4-connected region of unvisited cells, passing the
    // region's linear cell indices. `claimed` is scratch and must be zeroed by
    // the caller: it marks cells an earlier region already consumed. `stack`
    // and `cells` are reused between calls to avoid reallocating.
    template <class Fn>
    void ForEachUnvisitedHole(const MazeGenerator::VisitedGrid& visited,
                              MazeGenerator::VisitedGrid& claimed,
                              std::uint32_t width, std::uint32_t height,
                              std::vector<std::uint32_t>& stack,
                              std::vector<std::uint32_t>& cells, Fn&& on_hole) {
        constexpr std::size_t kCellsPerWord =
            MazeGenerator::VisitedGrid::kCellsPerWord;
        const std::size_t cell_count = visited.CellCount();

        for (std::size_t word = 0; word < visited.WordCount(); ++word) {
            const std::uint64_t visited_bits = visited.Word(word);
            if (visited_bits == ~std::uint64_t{0}) {
                continue; // one compare rejects 64 cells
            }

            // Padding bits past CellCount() in the last word are never written,
            // so they read as 0 ("unvisited") and would be picked as a seed.
            const std::size_t valid_bits =
                std::min(kCellsPerWord, cell_count - word * kCellsPerWord);
            const std::uint64_t valid =
                valid_bits == kCellsPerWord
                    ? ~std::uint64_t{0}
                    : (std::uint64_t{1} << valid_bits) - 1;

            std::uint64_t seeds = ~visited_bits & ~claimed.Word(word) & valid;
            while (seeds != 0) {
                const auto bit =
                    static_cast<std::uint32_t>(std::countr_zero(seeds));
                seeds &= seeds - 1;

                const auto seed =
                    static_cast<std::uint32_t>(word * kCellsPerWord + bit);
                if (claimed.Test(seed)) {
                    continue; // an earlier region reached it from another word
                }

                stack.clear();
                cells.clear();
                claimed.Set(seed, 1);
                stack.push_back(seed);
                cells.push_back(seed);

                while (!stack.empty()) {
                    const std::uint32_t index = stack.back();
                    stack.pop_back();
                    const std::uint32_t x = index % width;
                    const std::uint32_t y = index / width;

                    const auto visit = [&](std::uint32_t neighbour) {
                        if (!visited.Test(neighbour) &&
                            !claimed.Test(neighbour)) {
                            claimed.Set(neighbour, 1);
                            stack.push_back(neighbour);
                            cells.push_back(neighbour);
                        }
                    };
                    if (x > 0) {
                        visit(index - 1);
                    }
                    if (x + 1 < width) {
                        visit(index + 1);
                    }
                    if (y > 0) {
                        visit(index - width);
                    }
                    if (y + 1 < height) {
                        visit(index + width);
                    }
                }

                on_hole(std::span<const std::uint32_t>(cells));
            }
        }
    }

}

export namespace MazeGenerator {

// The walk's knobs: a multiplier on the base chance of stopping at a cell, the
// chance of branching instead of continuing straight, and how to deal with the
// unvisited rooms the walk leaves behind.
struct CustomGeneratorV1Settings {
    float stop_chance_scaler = 1.0f;
    float branch_chance = 0.2f;
    // A hole is only reopened when it holds more than this fraction of the
    // whole grid's cells. Smaller holes are left alone as rooms and nooks.
    float hole_fill_threshold = 0.2f;
    // Fraction of a reopened hole's border walls to open as doors.
    float hole_open_fraction = 0.075f;
};

inline void VisitCustomGeneratorV1Settings(
    CustomGeneratorV1Settings& settings, SettingsVisitor& visitor) {
    visitor.Float("Stop chance scaler", settings.stop_chance_scaler, 0.0f, 2.0f);
    visitor.Help("Multiplier on the base chance of stopping at a cell.");
    visitor.Float("Branch chance", settings.branch_chance, 0.0f, 1.0f);
    visitor.Help("Chance of branching instead of just turning.");
    // The useful band is a few thousandths to a few hundredths of the grid, so
    // the slider is logarithmic: a linear track would confine every value worth
    // picking to the first few pixels next to zero. The lower bound stays zero
    // because "reopen every hole" is a meaningful setting; dragging all the way
    // left still lands on exactly 0.0, and the four-decimal format only floors
    // the non-zero positions (0.0001 and up). It and `min` move together.
    visitor.Float("Hole Fill Threshold", settings.hole_fill_threshold, 0.0f,
                  1.0f, "%.4f", FieldHint::LogSlider);
    visitor.Help("Reopen holes holding more than this fraction of the grid; "
                 "smaller holes stay as rooms.");
    visitor.Float("Hole Open Fraction", settings.hole_open_fraction, 0.0f, 1.0f);
    visitor.Help("Fraction of a reopened hole's walls to open as doors.");
}

[[nodiscard]] inline std::optional<TiledMaze> GenerateCustomGeneratorV1(
    const MazeSpec& spec, const CustomGeneratorV1Settings& settings) {
    const std::uint32_t width = spec.width;
    const std::uint32_t height = spec.height;

    if (width == 0 || height == 0) {
        return std::nullopt;
    }

    const std::uint32_t averaged_size = (width + height) / 2;

    // Interior open, boundary closed; the walk enters and leaves only through
    // the two openings made below.
    TiledMaze maze = TiledMaze::Create(width, height, false);

    std::mt19937 rng(spec.seed);

    std::uniform_int_distribution<std::uint32_t> top_cell(0, width - 1);
    const std::uint32_t top_x = top_cell(rng);
    const std::uint32_t bottom_x = width - 1 - top_x;

    Position target = Position(bottom_x, height - 1);

    maze.OpenNorth(top_x, 0);             // top opening
    maze.OpenSouth(bottom_x, height - 1); // bottom opening

    // Scratch space for the carve.
    VisitedGrid visited = VisitedGrid::Create(width, height);

    std::queue<std::pair<Position, Direction>> open_branch_positions;

    // A hole is reopened only when it holds more cells than this.
    const double hole_fill_cells =
        static_cast<double>(width) * static_cast<double>(height) *
        static_cast<double>(settings.hole_fill_threshold);

    // Scratch for the hole pass, reused across rounds.
    VisitedGrid claimed = VisitedGrid::Create(width, height);
    std::vector<std::uint32_t> hole_stack;
    std::vector<std::uint32_t> hole_cells;

    // Called once the walk runs out of branch positions. Finds every unvisited
    // hole larger than the threshold, opens a fraction of its walls, and seeds
    // the branch queue inside it so generation resumes. Returns false when
    // nothing was seeded, which ends generation.
    auto TryOpenHolesAndResume = [&]() -> bool {
        claimed.Clear();
        bool seeded = false;
        std::size_t reopened = 0;
        std::size_t doors_opened = 0;

        ForEachUnvisitedHole(
            visited, claimed, width, height, hole_stack, hole_cells,
            [&](std::span<const std::uint32_t> hole) {
                if (static_cast<double>(hole.size()) <= hole_fill_cells) {
                    return;
                }
                ++reopened;

                HoleBorder border =
                    ClassifyHoleBorder(hole, visited, maze, width, height);
                const std::size_t perimeter =
                    border.passages.size() + border.walls.size();
                if (perimeter == 0) {
                    return; // no visited neighbour, so nothing to open from
                }

                // A sealed hole needs at least one door, otherwise the walk
                // can never get in.
                const float fraction =
                    std::clamp(settings.hole_open_fraction, 0.0f, 1.0f);
                std::size_t doors = static_cast<std::size_t>(
                    std::lround(fraction * static_cast<float>(perimeter)));
                if (border.passages.empty()) {
                    doors = std::max<std::size_t>(doors, 1);
                }
                doors = std::min(doors, border.walls.size());
                doors_opened += doors;

                std::shuffle(border.walls.begin(), border.walls.end(), rng);
                for (std::size_t i = 0; i < doors; ++i) {
                    const auto& [cell, dir] = border.walls[i];
                    OpenDirectionState(cell, dir, maze);
                    border.passages.push_back(border.walls[i]);
                }

                LOGIFACE_LOG(debug, std::format(
                    "hole of {} cells: {} passages, {} opened", hole.size(),
                    border.passages.size(), doors));

                for (const auto& [cell, door_dir] : border.passages) {
                    open_branch_positions.emplace(
                        cell,
                        InwardDirection(cell, visited, width, height, door_dir)
                            .value_or(Rotate(door_dir, 2)));
                    seeded = true;
                }
            });

        if (seeded) {
            LOGIFACE_LOG(info, std::format(
                "reopened {} hole(s), opened {} door(s)", reopened,
                doors_opened));
        }
        return seeded;
    };

    Position pos = Position(top_x, 0);
    Direction direction = Direction::left;

    visited.Set(pos.x, pos.y, 1);
    maze.CloseSouth(pos.x, pos.y);

    // Seed the first branch to the right of the entrance, but only when that
    // cell is part of the maze. The initial left step is guarded the same way.
    if (CanMove(pos, Direction::right, maze, width, height)) {
        LOGIFACE_LOG(debug, "adding branch position");
        open_branch_positions.emplace(MoveInDir(pos, Direction::right),
                                      Direction::right);
    }
    if (CanMove(pos, Direction::left, maze, width, height)) {
        pos = MoveInDir(pos, Direction::left);
    }

    std::uint32_t move_amount = 0;
    std::size_t jumps = 0;
    const std::size_t max_jumps =
        static_cast<std::size_t>(width) * height * (std::pow(averaged_size, 1.2f) / averaged_size);

    // Every seeded hole consumes at least one cell when the walk jumps into it,
    // so resuming is monotone and terminates; the cap only guards a logic error.
    constexpr std::size_t kMaxHoleRounds = 256;
    std::size_t hole_rounds = 0;

    bool continue_generating = true;
    while (continue_generating) {
        // The walk should terminate on its own; this only guards against a
        // wall-closing cycle growing the branch queue forever.
        if (jumps > max_jumps) {
            LOGIFACE_LOG(warn, "reached max jumps of " + std::to_string(max_jumps));
            break;
        }

        visited.Set(pos.x, pos.y, 1);

        const bool fullStop = !CanMove(pos, direction, maze, width, height);
        const Position target_pos = MoveInDir(pos, direction);
        const bool stop =
            fullStop || ShouldStop(settings.stop_chance_scaler, move_amount,
                                   averaged_size, rng) || visited.Get(target_pos.x, target_pos.y);

        if (!stop) {
            // Seal the sides of the cell we are leaving.
            // The cell where we stop or branch keeps its side passages open, which is what gives the walk somewhere to turn.
            // additionally, we should not seal if that cell is already visited.
            Position const right_cell = MoveInDir(pos, Rotate(direction, 1));
            Position const left_cell = MoveInDir(pos, Rotate(direction, -1));
            if (InBounds(right_cell, width, height) && !visited.Get(right_cell.x, right_cell.y)) {
                CloseDirectionState(pos, Rotate(direction, 1), maze);
            }
            if (InBounds(left_cell, width, height) && !visited.Get(left_cell.x, left_cell.y)) {
                CloseDirectionState(pos, Rotate(direction, -1), maze);
            }
            const Position temp_pos = MoveInDir(pos, direction);
            LOGIFACE_LOG(debug, "moving straight to " + std::to_string(temp_pos.x) + ", " + std::to_string(temp_pos.y) + " from " + std::to_string(pos.x) + ", " + std::to_string(pos.y));
            pos = temp_pos;
            ++move_amount;
            continue;
        }

        move_amount = 0;

        const bool should_attempt_branch = fullStop || ShouldBranch(settings.branch_chance, rng);

        const bool right_open =
            CanMoveIntoUnvisited(pos, Rotate(direction, 1), maze, visited, width,
                                 height);
        const bool left_open =
            CanMoveIntoUnvisited(pos, Rotate(direction, -1), maze, visited,
                                 width, height);

        if (!fullStop) {
            const Position target_pos = MoveInDir(pos, direction);
            if (InBounds(target_pos, width, height) && !visited.Get(target_pos.x, target_pos.y)) {
                CloseDirectionState(pos, direction, maze);
            }
        }

        if (should_attempt_branch && right_open && left_open) {
            // add the branched position, rotated right.
            LOGIFACE_LOG(debug, "adding branch position");
            std::int8_t turn_dir = GetFarthestRotationFromTarget(pos, target, direction);
            open_branch_positions.emplace(MoveInDir(pos, Rotate(direction, -turn_dir)),
                                          Rotate(direction, -turn_dir));
            direction = Rotate(direction, turn_dir);
            pos = MoveInDir(pos, direction);
        } else if (right_open || left_open) {
            const std::int8_t dir = right_open && left_open ? GetFarthestRotationFromTarget(pos, target, direction) : right_open ? 1 : -1;

            // Mirror the straight-move side seal: only close toward a cell the
            // walk has not visited. Sealing toward a visited cell can cut off
            // carved maze, and would close the door we just opened.
            const Position side_cell = MoveInDir(pos, Rotate(direction, -dir));
            if (InBounds(side_cell, width, height) &&
                !visited.Get(side_cell.x, side_cell.y)) {
                CloseDirectionState(pos, Rotate(direction, -dir), maze);
            }

            direction = Rotate(direction, dir);
            pos = MoveInDir(pos, direction);
        } else {
            bool advanced = false;
            while (!open_branch_positions.empty()) {
                const auto [branch_pos, branch_dir] = open_branch_positions.front();
                open_branch_positions.pop();

                if (visited.Get(branch_pos.x, branch_pos.y)) {
                    continue;  // already carved: jumping here re-enters the cycle
                }

                const bool has_exit =
                    CanMove(branch_pos, branch_dir, maze, width, height) ||
                    CanMove(branch_pos, Rotate(branch_dir, 1), maze, width, height) ||
                    CanMove(branch_pos, Rotate(branch_dir, -1), maze, width, height);
                if (!has_exit) {
                    continue;  // sealed while the walk was elsewhere
                }

                pos = branch_pos;
                LOGIFACE_LOG(debug, "moving to branch position " + std::to_string(pos.x) + ", " + std::to_string((pos.y)));
                direction = branch_dir;
                ++jumps;
                advanced = true;
                break;
            }
            if (!advanced) {
                // Out of branch positions. Reopen the big unvisited holes and
                // keep going; only give up when nothing is left to seed.
                if (hole_rounds >= kMaxHoleRounds) {
                    LOGIFACE_LOG(warn, "reached max hole rounds of " +
                                           std::to_string(kMaxHoleRounds));
                    continue_generating = false;
                } else {
                    ++hole_rounds;
                    continue_generating = TryOpenHolesAndResume();
                }
            }
        }
    }

    // The walk may close an entrance while passing over it. Re-open exactly
    // the two intended boundary openings before returning.
    maze.OpenNorth(top_x, 0);
    maze.OpenSouth(bottom_x, height - 1);

    if (!open_branch_positions.empty()) {
        LOGIFACE_LOG(warn, "open_branch_positions was not empty? it had " + std::to_string(open_branch_positions.size()) + " elements left");
    }

    LOGIFACE_LOG(debug, "jumped " + std::to_string(jumps) + " times.");

    return maze;
}

} // namespace MazeGenerator
