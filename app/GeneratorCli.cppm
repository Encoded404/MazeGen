// CLI surface for generator settings.
//
// Every registry entry's settings struct becomes real CLI11 options, namespaced by the generator's kebab-case name:
//
//   --random-open-probability 0.6
//   --custom-generator-v1-branch-chance 0.3
//
// The options bind directly into the type-erased settings values the caller owns, so make_defaults() followed by RegisterGeneratorOptions() and a parse yields the same concrete settings type the generator expects.
// Like the ImGui visitor, this one never names a concrete settings struct; adding a generator with one new registry row automatically gives it CLI flags.

module;

#include <CLI/CLI.hpp>
#include <cassert>

export module MazeGenerator.GeneratorCli;

import std;

import MazeGenerator.Generation;
import MazeGenerator.Generators;

export namespace MazeGenerator {

// "CustomGeneratorV1" -> "custom-generator-v1".
// This is the stable option prefix for a generator name; acronym runs stay together ("AStar" -> "a-star").
[[nodiscard]] std::string GeneratorOptionPrefix(std::string_view name) {
    std::string out;
    out.reserve(name.size() + 4);
    for (std::size_t i = 0; i < name.size(); ++i) {
        const char c = name[i];
        const bool upper = c >= 'A' && c <= 'Z';
        const bool lower = c >= 'a' && c <= 'z';
        const bool digit = c >= '0' && c <= '9';
        if (upper) {
            const bool word_start =
                i > 0 && ((name[i - 1] >= 'a' && name[i - 1] <= 'z') ||
                          (name[i - 1] >= '0' && name[i - 1] <= '9'));
            const bool acronym_end =
                i > 0 && name[i - 1] >= 'A' && name[i - 1] <= 'Z' &&
                i + 1 < name.size() && name[i + 1] >= 'a' &&
                name[i + 1] <= 'z';
            if (!out.empty() && (word_start || acronym_end)) {
                out += '-';
            }
            out += static_cast<char>(c - 'A' + 'a');
        } else if (lower || digit) {
            out += c;
        } else if (!out.empty() && out.back() != '-') {
            out += '-';
        }
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out;
}

// Registers one CLI11 option per settings field, bound straight to the value.
// `app` is usually a per-generator option group.
// Option names are `--<prefix>-<group path>-<key>`, so the same key can exist in different generators.
// Hyphens rather than dots are deliberate: CLI11 treats dots in a command-line token as subcommand dot-notation, so `--random.open-probability` would not parse as a plain option.
class CliSettingsVisitor final : public SettingsVisitor {
public:
    CliSettingsVisitor(CLI::App& app, std::string prefix)
        : current_(&app), prefix_(std::move(prefix)) {}

    void Float(const SettingsField& field, float& value, float min, float max,
               const char* /*format*/, FieldHint /*hint*/) override {
        last_ = current_->add_option(OptionName(field.key), value, field.label)
                    ->check(CLI::Range(static_cast<double>(min),
                                       static_cast<double>(max)))
                    ->type_name("F")
                    ->capture_default_str();
        last_label_ = field.label;
    }

    void Int(const SettingsField& field, int& value, int min, int max,
             FieldHint /*hint*/) override {
        last_ = current_->add_option(OptionName(field.key), value, field.label)
                    ->check(CLI::Range(min, max))
                    ->type_name("N")
                    ->capture_default_str();
        last_label_ = field.label;
    }

    void Bool(const SettingsField& field, bool& value) override {
        last_ = current_->add_flag(OptionName(field.key), value, field.label);
        last_label_ = field.label;
    }

    void Enum(const SettingsField& field, int& value,
              std::span<const EnumOption> options) override {
        std::map<std::string, std::string> names;
        for (const EnumOption& option : options) {
            names.emplace(option.label, std::to_string(option.value));
        }
        last_ = current_->add_option(OptionName(field.key), value, field.label)
                    ->transform(CLI::CheckedTransformer(std::move(names),
                                                        CLI::ignore_case))
                    ->type_name("NAME")
                    ->capture_default_str();
        last_label_ = field.label;
    }

    void Path(const SettingsField& field, std::string& value) override {
        last_ = current_->add_option(OptionName(field.key), value, field.label)
                    ->type_name("PATH")
                    ->capture_default_str();
        last_label_ = field.label;
    }

    bool BeginGroup(const SettingsField& group, bool /*default_open*/) override {
        groups_.push_back({.parent = current_, .path_size = group_path_.size()});
        current_ = current_->add_option_group(group.label);
        if (!group_path_.empty()) {
            group_path_ += '-';
        }
        group_path_ += group.key;
        last_ = nullptr;
        return true;
    }

    void EndGroup() override {
        if (groups_.empty()) {
            return;
        }
        current_ = groups_.back().parent;
        group_path_.resize(groups_.back().path_size);
        groups_.pop_back();
        last_ = nullptr;
    }

    void Help(std::string_view text) override {
        if (last_ == nullptr || text.empty()) {
            return;
        }
        std::string description = last_label_;
        if (!description.empty()) {
            description += ". ";
        }
        description += text;
        last_->description(description);
    }

private:
    struct GroupFrame {
        CLI::App* parent = nullptr;
        std::size_t path_size = 0;
    };

    [[nodiscard]] std::string OptionName(std::string_view key) const {
        std::string name = "--";
        name += prefix_;
        if (!group_path_.empty()) {
            name += '-';
            name += group_path_;
        }
        name += '-';
        name += key;
        return name;
    }

    CLI::App* current_ = nullptr;
    std::string prefix_;
    std::string group_path_;
    std::vector<GroupFrame> groups_;
    CLI::Option* last_ = nullptr;
    std::string last_label_;
};

// Registers the settings of every generator in `generators`.
// `settings` holds one value per entry, in the same order, created with make_defaults(); the values are modified in place during parsing.
// The caller must keep them alive (and not reallocate) until parsing finishes, because CLI11 stores references.
void RegisterGeneratorOptions(CLI::App& app,
                              std::span<const MazeGeneratorType> generators,
                              std::span<std::any> settings) {
    assert(generators.size() == settings.size());
    const std::size_t count = std::min(generators.size(), settings.size());
    for (std::size_t i = 0; i < count; ++i) {
        const MazeGeneratorType& generator = generators[i];
        CLI::App* group = app.add_option_group(std::string(generator.name));
        CliSettingsVisitor visitor(*group, GeneratorOptionPrefix(generator.name));
        generator.visit(settings[i], visitor);
    }
}

} // namespace MazeGenerator
