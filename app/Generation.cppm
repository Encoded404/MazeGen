// Maze generation contract.
//
// A generator is a pure function: maze dimensions and a seed in, a TiledMaze
// out. Each generator owns a concrete settings struct, and describes it through
// a SettingsVisitor so the UI can render the selected generator's controls
// without knowing its settings type. The module deliberately has no ImGui or
// Vulkan dependency so generators stay testable headless; an ImGui visitor
// lives in the app and a future loader can provide a JSON visitor.

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

// How a numeric field should be presented.
// Input accepts free-form entry.
// Drag is a scrubbable field.
// Slider presents a track over the declared range, which both ImGui and the CLI treat as soft limits.
// LogSlider is a Slider whose track is logarithmic: use it for settings that live in a small band near one end of their range, where a linear track would spend the whole width on values nobody picks.
// Two caveats come with it, both from the underlying widget.
// First, the format's decimal count sets the smallest non-zero value the track can reach (`"%.4f"` bottoms out at 0.0001), so choose `min` and `format` together rather than relying on the range alone.
// Second, a `min` of zero stays reachable: the far-left end of the track yields exactly zero, and only the positions above it are log-spaced from that 10^-digits floor upward.
enum class FieldHint : std::uint8_t {
    Slider,
    Drag,
    Input,
    LogSlider,
};

// One entry in an enum combo. Values cross the visitor boundary as int so the
// interface does not need a template per enum type; VisitEnum converts back.
struct EnumOption {
    const char* label = "";
    int value = 0;
};

// A setting's identity.
// `key` is the stable machine name used by CLI flags, config files, and tests; `label` is the display name the UI renders.
// Rewording a label must not change a key, and keys must be unique within one generator.
struct SettingsField {
    const char* key = "";
    const char* label = "";
};

// Walks one generator's settings.
// Implementations live in the consumer:
// the app provides an ImGui one and a CLI11 one (GeneratorCli).
// tests provide a recording one.
// a future settings loader can provide a JSON one.
// Each generator exports a Visit<Name>Settings(Settings&, SettingsVisitor&) function beside its settings struct.
class SettingsVisitor {
public:
    virtual ~SettingsVisitor() = default;

    virtual void Float(const SettingsField& field, float& value, float min,
                       float max, const char* format = "%.2f",
                       FieldHint hint = FieldHint::Slider) = 0;
    virtual void Int(const SettingsField& field, int& value, int min, int max,
                     FieldHint hint = FieldHint::Slider) = 0;
    virtual void Bool(const SettingsField& field, bool& value) = 0;
    virtual void Enum(const SettingsField& field, int& value,
                      std::span<const EnumOption> options) = 0;
    virtual void Path(const SettingsField& field, std::string& value) = 0;

    // A group is a collapsible section (a "folder" in the panel). Returns true
    // when the visitor wants the children, in which case the caller must visit
    // them and call EndGroup(). Returns false when the section is collapsed, in
    // which case the children are skipped and EndGroup() is not called. Groups
    // nest.
    virtual bool BeginGroup(const SettingsField& group,
                            bool default_open = true) = 0;
    virtual void EndGroup() = 0;

    // Attaches a tooltip to the widget most recently declared.
    virtual void Help(std::string_view text) = 0;
};

// Binds a generator's enum to the int-based visitor interface and writes the
// selection back. The enum must have an int-compatible underlying type; a
// visitor that reports a value outside `options` leaves the setting unchanged.
template <class Enum>
void VisitEnum(SettingsVisitor& visitor, const SettingsField& field, Enum& value,
               std::span<const EnumOption> options) {
    static_assert(std::is_enum_v<Enum>, "VisitEnum requires an enum type");
    int raw = static_cast<int>(value);
    visitor.Enum(field, raw, options);
    for (const EnumOption& option : options) {
        if (option.value == raw) {
            value = static_cast<Enum>(raw);
            return;
        }
    }
}

} // namespace MazeGenerator
