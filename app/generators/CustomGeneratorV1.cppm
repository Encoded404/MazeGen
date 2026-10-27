// CustomGeneratorV1 maze generator scaffold.
//
// The generation algorithm is intentionally not implemented yet. The module,
// registry entry, and build wiring are in place so the algorithm can be added
// here without touching the app, the UI, or the CMake module lists again.

module;

export module MazeGenerator.Generators.CustomGeneratorV1;

import std;

import MazeGenerator.Generation;
import MazeGenerator.TiledMaze;

enum class Direction : std::uint8_t {
    up = 0,
    right = 1,
    down = 2,
    left = 3,
};

export namespace MazeGenerator {

// The walk's two knobs: a multiplier on the base chance of stopping at a cell,
// and the chance of branching instead of continuing straight.
struct CustomGeneratorV1Settings {
    float stop_chance_scaler = 1.0f;
    float branch_chance = 0.2f;
};

inline void VisitCustomGeneratorV1Settings(
    CustomGeneratorV1Settings& settings, SettingsVisitor& visitor) {
    visitor.Float("Stop chance scaler", settings.stop_chance_scaler, 0.0f, 2.0f);
    visitor.Help("Multiplier on the base chance of stopping at a cell.");
    visitor.Float("Branch chance", settings.branch_chance, 0.0f, 1.0f);
    visitor.Help("Chance of branching instead of continuing straight.");
}

[[nodiscard]] inline std::optional<TiledMaze> GenerateCustomGeneratorV1(
    const MazeSpec& spec, const CustomGeneratorV1Settings& settings) {
    const std::uint32_t width = spec.width;
    const std::uint32_t height = spec.height;

    if (width == 0 || height == 0) {
        return std::nullopt;
    }

    TiledMaze maze = TiledMaze::Create(width, height);

    std::mt19937 rng(spec.seed);

    std::uniform_int_distribution<std::uint32_t> top_cell(0, width - 1);
    const std::uint32_t top_x = top_cell(rng);
    const std::uint32_t bottom_x = width - 1 - top_x;

    maze.OpenNorth(top_x, 0);             // top opening
    maze.OpenSouth(bottom_x, height - 1); // bottom opening

    [[maybe_unused]] std::uint32_t x = top_x;      // start at the entrance.
    [[maybe_unused]] std::uint32_t y = 0;
    [[maybe_unused]] Direction direction = Direction::down;

    (void)settings;
    // TODO: implement the CustomGeneratorV1 algorithm.
    // Until then, report that these settings cannot produce a maze so the app
    // keeps its previous maze.
    return std::nullopt;
}

} // namespace MazeGenerator
