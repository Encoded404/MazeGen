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

export namespace MazeGenerator {

[[nodiscard]] inline std::optional<TiledMaze> GenerateCustomGeneratorV1(const MazeSpec& spec, const MazeGenSettings& settings) {
    (void)spec;
    (void)settings;
    // TODO: implement the CustomGeneratorV1 algorithm.
    // Until then, report that these settings cannot produce a maze so the app
    // keeps its previous maze.
    return std::nullopt;
}

} // namespace MazeGenerator
