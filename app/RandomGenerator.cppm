// The random-wall generator: every interior wall is independently open with
// `open_probability`, plus one top and one mirrored bottom opening.

module;

export module MazeGenerator.Generators.Random;

import std;

import MazeGenerator.Generation;
import MazeGenerator.TiledMaze;

export namespace MazeGenerator {

[[nodiscard]] inline std::optional<TiledMaze> GenerateRandomMaze(
    const MazeSpec& spec, const MazeGenSettings& settings) {
    if (spec.width == 0 || spec.height == 0) {
        return std::nullopt;
    }
    return TiledMaze::CreateRandom(
        spec.width, spec.height,
        std::clamp(static_cast<double>(settings.open_probability), 0.0, 1.0),
        spec.seed);
}

} // namespace MazeGenerator
