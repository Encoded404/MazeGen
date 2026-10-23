module;

export module Runtime.Overrides;

import std;

import VulkanEngine.GplPolicy;
import VulkanEngine.DrawMode;

export namespace Runtime {

// Re-export the engine policy types so importers of this module can name them
// without importing engine internals themselves. Plain using-declarations are
// exported because they sit in an exported namespace.
using VulkanEngine::ShaderSystem::GplPolicy;
using VulkanEngine::ShaderSystem::GplStructurePolicy;
using VulkanEngine::SceneRenderer::DrawMode;

// One possible value for a typed override. Parse, hint, help and format all
// derive from this table — the value grammar lives exactly once.
template <typename T>
struct Choice {
    std::string_view label;        // CLI spelling, e.g. "force"
    T value;                       // typed value it maps to
    std::string_view description;  // one-liner shown in --help
};

class OverrideError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Overrides;

struct OverrideSpec {
    std::string_view key;                // "ext.disable" | "gpl"
    std::string_view summary;            // one-line help
    std::string_view details;            // optional extended text (may be empty)
    bool repeatable = false;             // shown in help as [repeatable]
    std::string (*value_hint)();                  // "<extension name>" | generated "auto|force|off"
    void (*apply)(std::string_view value, Overrides&);  // validate + dispatch; throws OverrideError
    std::string (*describe)(const Overrides&);          // format current value for help
    std::string (*choices_help)() = nullptr;            // optional: per-choice lines for --help
};

// Built-in override choices.
inline constexpr std::array kGplChoices = std::to_array<Choice<GplPolicy>>({
    {"auto",  GplPolicy::Auto,    "engine policy (RADV workaround applies)"},
    {"force", GplPolicy::Forced,  "bypass GPL policy restrictions, e.g. combined structure on affected RADV; for testing"},
    {"off",   GplPolicy::Disable, "always use monolithic pipelines"},
});

inline constexpr std::array kGplStructureChoices = std::to_array<Choice<GplStructurePolicy>>({
    {"auto",     GplStructurePolicy::Auto,     "engine policy (RADV < Mesa 26 → split, otherwise combined)"},
    {"split",    GplStructurePolicy::Split,    "separate FS/FOI libraries; required on RADV < Mesa 26 (doc §5)"},
    {"combined", GplStructurePolicy::Combined, "single combined FS+FOI library; broken on RADV < Mesa 26 (doc §3.1/§4)"},
});

// Draw-mode override. std::nullopt ("auto") means "no override": the game or
// settings default applies. The engine enum has no Auto value, so the optional
// carries that neutral third state. This keeps an unset override from
// overwriting a mode chosen elsewhere (a future settings UI).
inline constexpr std::array kDrawModeChoices = std::to_array<Choice<std::optional<DrawMode>>>({
    {"auto", std::nullopt,  "no override; use the game or settings draw mode"},
    {"cid",  DrawMode::CID, "one drawIndexedIndirect per pass/technique; core 1.0 only"},
    {"mid",  DrawMode::MID, "one command per alive submesh; needs drawIndirectCount + drawIndirectFirstInstance"},
});

// Override registry. The engine-standard keys (ext.disable, gpl, gpl.structure,
// draw.mode) are built in; examples extend the registry with Register() to add
// their own --overwrite keys. Apply must be called after all registrations.
class Overrides {
public:
    // Built-in outcome state, consumed directly by ApplicationConfig / GameConfig.
    std::vector<std::string> force_disabled_extensions;
    GplPolicy gpl_policy = GplPolicy::Auto;
    GplStructurePolicy gpl_structure = GplStructurePolicy::Auto;
    std::optional<DrawMode> draw_mode = std::nullopt;

    // Register an app-defined --overwrite key. Must be called before
    // Apply()/GenerateHelp() so parse and help see it.
    void Register(OverrideSpec spec);

    // Dispatch "key=value"; throws OverrideError for malformed or unknown keys.
    void Apply(std::string_view kv);

    // Aligned help text for the --help footer (built-ins + registered extras).
    std::string GenerateHelp() const;

    // One-line "<key>=<value>, ..." summary for the effective-overrides log.
    std::string DescribeEffective() const;

private:
    std::vector<OverrideSpec> extra_specs_{};
};

} // namespace Runtime