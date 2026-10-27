// The generator registry. Adding a generator is a new .cppm under
// app/generators/ with its own settings struct, visit function, and generate
// function plus one MakeGenerator entry in kMazeGenerators; the UI and the App
// read the table and never name a concrete settings type.

module;

export module MazeGenerator.Generators;

import std;

import MazeGenerator.Generation;
import MazeGenerator.Generators.CustomGeneratorV1;
import MazeGenerator.Generators.Random;
import MazeGenerator.TiledMaze;

export namespace MazeGenerator {

// Type-erased view of one generator's settings. The thunks below are
// instantiated for the generator's concrete settings struct, so generation and
// the UI stay typed inside the generator while the registry remains a
// homogeneous table. The app owns one std::any per entry, created from
// make_defaults(), which is what makes settings per generator.
struct MazeGeneratorType {
    std::string_view name;
    std::optional<TiledMaze> (*generate)(const MazeSpec&, const std::any&) = nullptr;
    void (*visit)(std::any&, SettingsVisitor&) = nullptr;
    std::any (*make_defaults)() = nullptr;
};

// One specialization per registry entry. Function pointers travel as template
// parameters because a captureless lambda cannot capture the `generate` and
// `visit` arguments passed to MakeGenerator.
template <class Settings, auto Generate, auto Visit>
struct GeneratorThunks {
    static std::optional<TiledMaze> GenerateErased(const MazeSpec& spec,
                                                   const std::any& settings) {
        return Generate(spec, std::any_cast<const Settings&>(settings));
    }

    static void VisitErased(std::any& settings, SettingsVisitor& visitor) {
        Visit(std::any_cast<Settings&>(settings), visitor);
    }

    static std::any MakeDefaults() { return std::any{Settings{}}; }
};

template <class Settings, auto Generate, auto Visit>
consteval MazeGeneratorType MakeGenerator(std::string_view name) {
    using Thunks = GeneratorThunks<Settings, Generate, Visit>;
    return MazeGeneratorType{
        .name = name,
        .generate = &Thunks::GenerateErased,
        .visit = &Thunks::VisitErased,
        .make_defaults = &Thunks::MakeDefaults,
    };
}

inline constexpr MazeGeneratorType kMazeGenerators[] = {
    MakeGenerator<RandomSettings, &GenerateRandomMaze, &VisitRandomSettings>(
        "Random"),
    MakeGenerator<CustomGeneratorV1Settings, &GenerateCustomGeneratorV1,
                  &VisitCustomGeneratorV1Settings>("CustomGeneratorV1"),
};

[[nodiscard]] inline std::span<const MazeGeneratorType> MazeGeneratorTypes() noexcept {
    return kMazeGenerators;
}

} // namespace MazeGenerator
