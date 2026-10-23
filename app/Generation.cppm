// Maze generation contract.
//
// A generator is a pure function: maze dimensions and a seed in, a TiledMaze
// out. Parameter metadata lives beside the function so the UI can render the
// selected generator's controls without knowing which generator it is. The
// module deliberately has no ImGui or Vulkan dependency so generators stay
// testable headless.

module;

export module MazeGenerator.Generation;

import std;

import MazeGenerator.TiledMaze;

export namespace MazeGenerator {

// What the caller asks for, independent of the algorithm that builds it.
struct MazeSpec {
    std::uint32_t width = 4096;
    std::uint32_t height = 4096;
    std::uint32_t seed = 0;
};

// One settings block shared by every generator. Each generator declares the
// fields it reads through its descriptor table, so a new knob is one field plus
// one table row. If a future generator needs structured parameters (a list of
// rooms, a mask image), give that generator its own settings object instead of
// growing this one.
struct MazeGenSettings {
    float open_probability = 0.45f; // Random
};

// A float knob rendered as SliderFloat. `value` is a pointer-to-member so the
// UI can bind it without a switch over generator kinds.
struct FloatParam {
    const char* label = "";
    float MazeGenSettings::* value = nullptr;
    float min = 0.0f;
    float max = 1.0f;
    const char* format = "%.2f";
};

// Integer counterpart, used by generators that need counts.
struct IntParam {
    const char* label = "";
    int MazeGenSettings::* value = nullptr;
    int min = 0;
    int max = 100;
};

// nullopt means "the current settings cannot produce a maze"; the caller keeps
// its previous maze and reports the failure.
using MazeGenerateFn =
    std::optional<TiledMaze> (*)(const MazeSpec&, const MazeGenSettings&);

} // namespace MazeGenerator
