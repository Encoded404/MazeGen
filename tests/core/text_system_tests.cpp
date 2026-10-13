#include <gtest/gtest.h>

import std;
import std.compat;

import FileLoader.Types;
import VulkanEngine.Render.Passes.WorldTextPass;
import VulkanEngine.ResourceSystem;
import VulkanEngine.ResourceSystem.FontResource;
import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;
import VulkanEngine.Text.FontReloader;
import VulkanEngine.Text.FontWatcher;
import VulkanEngine.Text.GlyphRaster;
import VulkanEngine.Text.Shaping;
import VulkanEngine.Text.TextSystem;

namespace {

using VulkanEngine::FontResource;
using VulkanEngine::ResourceId;
using VulkanEngine::Text::FontReloader;
using VulkanEngine::Text::FontWatcher;
using VulkanEngine::Text::TextSystem;

[[nodiscard]] std::filesystem::path TestFontPath() {
    return std::filesystem::path{VKENGINE_TEST_FONT_DIR} / "Lato-Regular.ttf";
}

[[nodiscard]] std::vector<std::byte> ReadFileBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return {};
    }
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        return {};
    }
    return bytes;
}

class TempFontFile final {
public:
    explicit TempFontFile(std::string_view extension) {
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        std::random_device rd;
        path_ = std::filesystem::temp_directory_path() /
                ("ve5_text_system_" + std::to_string(stamp) + "_" + std::to_string(rd()) +
                 std::string(extension));
    }

    ~TempFontFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& Path() const noexcept { return path_; }

    [[nodiscard]] bool Write(std::span<const std::byte> bytes) const {
        std::ofstream file(path_, std::ios::binary | std::ios::trunc);
        if (!file) {
            return false;
        }
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        return static_cast<bool>(file);
    }

private:
    std::filesystem::path path_;
};

// The same (resource, face index) is opened once: the second request hands back
// the same face object, not a second copy of the same font.
TEST(TextSystemTest, SameFontRequestedTwiceYieldsTheSameFace) {
    const std::vector<std::byte> bytes = ReadFileBytes(TestFontPath());
    ASSERT_FALSE(bytes.empty());

    TextSystem system;
    FontResource resource(ResourceId{.value = "lato"});
    ASSERT_TRUE(resource.Load(FileLoader::ByteBuffer{bytes}));

    const auto first = system.LoadFont(ResourceId{.value = "lato"}, resource);
    const auto second = system.LoadFont(ResourceId{.value = "lato"}, resource);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first.get(), second.get());
    EXPECT_EQ(system.FontCount(), 1u);
}

// Two different fonts are two different faces even when their bytes are
// identical: identity is the resource, and the UniqueId proves it.
TEST(TextSystemTest, DifferentFontsYieldDifferentFaces) {
    const std::vector<std::byte> bytes = ReadFileBytes(TestFontPath());
    ASSERT_FALSE(bytes.empty());

    TextSystem system;
    FontResource one(ResourceId{.value = "font-one"});
    FontResource two(ResourceId{.value = "font-two"});
    ASSERT_TRUE(one.Load(FileLoader::ByteBuffer{bytes}));
    ASSERT_TRUE(two.Load(FileLoader::ByteBuffer{bytes}));

    const auto first = system.LoadFont(ResourceId{.value = "font-one"}, one);
    const auto second = system.LoadFont(ResourceId{.value = "font-two"}, two);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_NE(first->UniqueId(), second->UniqueId());
    EXPECT_EQ(system.FontCount(), 2u);
}

// A face handed out before a reload stays valid and usable after the registry
// entry has been replaced: the registry holds it by shared_ptr, and so does the
// caller.
TEST(TextSystemTest, AnEarlierFaceStaysAliveAndUsableAcrossAReload) {
    const std::vector<std::byte> bytes = ReadFileBytes(TestFontPath());
    ASSERT_FALSE(bytes.empty());

    std::vector<std::byte> padded = bytes;
    padded.insert(padded.end(), 8, std::byte{0x00});
    const TempFontFile file(".ttf");
    ASSERT_TRUE(file.Write(padded));

    TextSystem system;
    FontResource resource(ResourceId{.value = "lato"});
    ASSERT_TRUE(resource.Load(FileLoader::ByteBuffer{bytes}));

    const auto old_face = system.LoadFont(ResourceId{.value = "lato"}, resource, 0, file.Path());
    ASSERT_NE(old_face, nullptr);
    const std::uint64_t old_id = old_face->UniqueId();

    ASSERT_TRUE(system.ReloadFont(ResourceId{.value = "lato"}, file.Path(), /*frame_index=*/0));

    EXPECT_EQ(old_face->UniqueId(), old_id);
    EXPECT_GE(old_face.use_count(), 1);
    EXPECT_GT(old_face->Metrics().units_per_em, 0u);
    EXPECT_NE(old_face->GlyphForCodepoint(static_cast<std::uint32_t>('H')), 0u);

    const auto new_face = system.GetFace(ResourceId{.value = "lato"});
    ASSERT_NE(new_face, nullptr);
    EXPECT_NE(new_face.get(), old_face.get());
    EXPECT_NE(new_face->UniqueId(), old_id);
    EXPECT_EQ(new_face->ResourceVersion(), 2u);
}

// A valid reload invalidates everything derived from the replaced face: the
// shaping cache stops serving it (the new face is a different key), the atlas no
// longer holds its glyph keys, and a run already handed out stays readable.
TEST(TextSystemTest, ReloadInvalidatesTheOldFacesCachesAndAtlas) {
    const std::vector<std::byte> bytes = ReadFileBytes(TestFontPath());
    ASSERT_FALSE(bytes.empty());

    std::vector<std::byte> padded = bytes;
    padded.insert(padded.end(), 8, std::byte{0x00});
    const TempFontFile file(".ttf");
    ASSERT_TRUE(file.Write(padded));

    TextSystem system;
    FontResource resource(ResourceId{.value = "lato"});
    ASSERT_TRUE(resource.Load(FileLoader::ByteBuffer{bytes}));
    const auto face = system.LoadFont(ResourceId{.value = "lato"}, resource, 0, file.Path());
    ASSERT_NE(face, nullptr);

    const auto old_run = system.GetShapingCache().Shape(*face, "Hey");
    ASSERT_NE(old_run, nullptr);
    EXPECT_EQ(system.GetShapingCache().Size(), 1u);

    const std::uint32_t glyph = face->GlyphForCodepoint(static_cast<std::uint32_t>('H'));
    ASSERT_NE(glyph, 0u);
    ASSERT_TRUE(system.GetRasterizer().Rasterize(*face, glyph, 24.0f).has_value());
    ASSERT_GT(system.GetRasterizer().Atlas().GlyphCount(), 0u);

    const std::uint64_t old_id = face->UniqueId();
    ASSERT_TRUE(system.ReloadFont(ResourceId{.value = "lato"}, file.Path(), /*frame_index=*/0));
    const auto new_face = system.GetFace(ResourceId{.value = "lato"});
    ASSERT_NE(new_face, nullptr);
    ASSERT_NE(new_face->UniqueId(), old_id);

    // Shaping: the old face's entry was dropped, so shaping it again is a fresh
    // shape (no hit) and hands back a different run object.
    EXPECT_EQ(system.GetShapingCache().Size(), 0u);
    const std::uint64_t hits_before = system.GetShapingCache().HitCount();
    const auto re_shaped = system.GetShapingCache().Shape(*face, "Hey");
    ASSERT_NE(re_shaped, nullptr);
    EXPECT_EQ(system.GetShapingCache().HitCount(), hits_before)
        << "the old face's run must have been invalidated, not served";
    EXPECT_NE(re_shaped.get(), old_run.get());

    // The previously returned run is still readable.
    ASSERT_FALSE(old_run->glyphs.empty());
    EXPECT_FALSE(old_run->text.empty());

    // Glyphs and atlas: the rasterizer was reset, so no page holds the old
    // face's rectangles any more.
    EXPECT_EQ(system.GetRasterizer().Size(), 0u);
    EXPECT_EQ(system.GetRasterizer().Atlas().GlyphCount(), 0u);
}

// A rejected reload changes nothing: the face keeps its identity, the resource
// keeps its version, and the caches stay as they were.
TEST(TextSystemTest, RejectedReloadLeavesTheFaceAndCachesUntouched) {
    const std::vector<std::byte> bytes = ReadFileBytes(TestFontPath());
    ASSERT_FALSE(bytes.empty());

    const std::vector<std::byte> junk(128, std::byte{0x33});
    const TempFontFile garbage(".ttf");
    ASSERT_TRUE(garbage.Write(junk));

    TextSystem system;
    FontResource resource(ResourceId{.value = "lato"});
    ASSERT_TRUE(resource.Load(FileLoader::ByteBuffer{bytes}));
    const auto face = system.LoadFont(ResourceId{.value = "lato"}, resource, 0, garbage.Path());
    ASSERT_NE(face, nullptr);
    const std::uint64_t old_id = face->UniqueId();
    ASSERT_NE(system.GetShapingCache().Shape(*face, "Hey"), nullptr);

    EXPECT_FALSE(system.ReloadFont(ResourceId{.value = "lato"}, garbage.Path(), 0));
    EXPECT_EQ(resource.GetVersion(), 1u);
    EXPECT_EQ(system.GetFace(ResourceId{.value = "lato"})->UniqueId(), old_id);
    EXPECT_EQ(system.GetShapingCache().Size(), 1u);
}

// Reloading a font that was never registered is a no-op, not a crash.
TEST(TextSystemTest, ReloadingAnUnknownFontIsANoOp) {
    TextSystem system;
    EXPECT_FALSE(
        system.ReloadFont(ResourceId{.value = "not-registered"}, TestFontPath(), /*frame=*/0));
    EXPECT_EQ(system.GetFace(ResourceId{.value = "not-registered"}), nullptr);
}

// The reloader turns a watched file's change into a reload: a notified path
// matches the registration and the registry entry is replaced.
TEST(TextSystemTest, FontReloaderReloadsAChangedFile) {
    const std::vector<std::byte> bytes = ReadFileBytes(TestFontPath());
    ASSERT_FALSE(bytes.empty());
    const TempFontFile file(".ttf");
    ASSERT_TRUE(file.Write(bytes));

    VulkanEngine::ResourceManager manager;
    TextSystem system;
    system.SetResourceManager(manager);
    const ResourceId id{file.Path().string()};
    const auto before = system.LoadFontFromPath(file.Path());
    ASSERT_NE(before, nullptr);

    FontWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    FontReloader reloader;
    reloader.Initialize(watcher, system);
    reloader.WatchResource(id, file.Path());
    EXPECT_EQ(reloader.WatchedCount(), 1u);
    EXPECT_EQ(watcher.GetRegisteredDirectoryCount(), 1u);

    watcher.NotifyChanged(file.Path().string());
    EXPECT_EQ(reloader.Pump(/*frame_index=*/0), 1u);

    const auto after = system.GetFace(id);
    ASSERT_NE(after, nullptr);
    EXPECT_NE(after->UniqueId(), before->UniqueId());
    EXPECT_EQ(after->ResourceVersion(), 2u);
    reloader.Shutdown();
    watcher.Stop();
}

// A change to an unwatched path never reloads anything.
TEST(TextSystemTest, FontReloaderIgnoresUnwatchedPaths) {
    const std::vector<std::byte> bytes = ReadFileBytes(TestFontPath());
    ASSERT_FALSE(bytes.empty());
    const TempFontFile file(".ttf");
    ASSERT_TRUE(file.Write(bytes));

    VulkanEngine::ResourceManager manager;
    TextSystem system;
    system.SetResourceManager(manager);
    const ResourceId id{file.Path().string()};
    const auto face = system.LoadFontFromPath(file.Path());
    ASSERT_NE(face, nullptr);
    const std::uint64_t face_id = face->UniqueId();

    FontWatcher watcher;
    watcher.SetDebounce(std::chrono::milliseconds(0));
    FontReloader reloader;
    reloader.Initialize(watcher, system);
    reloader.WatchResource(id, file.Path());

    watcher.NotifyChanged("/tmp/ve5_some_other_font.ttf");
    EXPECT_EQ(reloader.Pump(/*frame_index=*/0), 0u);
    EXPECT_EQ(system.GetFace(id)->UniqueId(), face_id);
    reloader.Shutdown();
    watcher.Stop();
}

// No-flag runs keep historical MSDF behavior — backward-compat contract. The
// engine's documented default world-text rasterizer must not silently flip to
// Slug (a different sampler and instance layout) without a deliberate update
// here; every run that does not name a backend inherits this value through
// RendererConfig.
TEST(TextBackendDefaultTest, NoFlagRunsKeepHistoricalMsdfBehavior) {
    using VulkanEngine::SceneRenderer::kDefaultTextBackend;
    using VulkanEngine::SceneRenderer::TextBackend;
    EXPECT_EQ(TextBackend::Msdf, kDefaultTextBackend);
}

}  // namespace
