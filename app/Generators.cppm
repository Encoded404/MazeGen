// The generator registry. Adding a generator is a new .cppm with a generate
// function plus one entry in kMazeGenerators; the UI and the App read the table.

module;

export module MazeGenerator.Generators;

import std;

import MazeGenerator.Generation;
import MazeGenerator.Generators.Random;

export namespace MazeGenerator {

struct MazeGeneratorType {
    std::string_view name;
    MazeGenerateFn generate = nullptr;
    std::span<const FloatParam> floats{};
    std::span<const IntParam> ints{};
};

inline constexpr FloatParam kRandomParams[] = {
    {"Open probability", &MazeGenSettings::open_probability, 0.0f, 1.0f, "%.2f"},
};

inline constexpr MazeGeneratorType kMazeGenerators[] = {
    {"Random", &GenerateRandomMaze, std::span<const FloatParam>{kRandomParams}, {}},
};

[[nodiscard]] inline std::span<const MazeGeneratorType> MazeGeneratorTypes() noexcept {
    return kMazeGenerators;
}

} // namespace MazeGenerator
