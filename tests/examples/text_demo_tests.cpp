#include <gtest/gtest.h>

import std;

import Examples.TextDemo.Config;

namespace {

using Examples::TextDemo::MakeRendererConfig;
using Examples::TextDemo::ParseTextBackend;
using Examples::TextDemo::TextBackendName;
using VulkanEngine::Renderer::RendererConfig;
using VulkanEngine::SceneRenderer::kDefaultTextBackend;
using VulkanEngine::SceneRenderer::TextBackend;
using VulkanEngine::SceneRenderer::WorldTextPass;

// The example's whole device-free surface is the configuration path: the
// command-line spelling is parsed once and mapped onto RendererConfig, and that
// value decides which shader pair (and which instance layout) the engine's one
// world-text pass is built with. Everything past that needs a device; the
// engine's own world-text and text-system GPU tests cover the drawing.

TEST(TextDemoConfigTest, ParsesBackendNamesCaseInsensitively) {
    EXPECT_EQ(ParseTextBackend("msdf"), TextBackend::Msdf);
    EXPECT_EQ(ParseTextBackend("MSDF"), TextBackend::Msdf);
    EXPECT_EQ(ParseTextBackend("slug"), TextBackend::Slug);
    EXPECT_EQ(ParseTextBackend("Slug"), TextBackend::Slug);

    // An unknown spelling is rejected rather than silently resolved to the
    // default: the backend is the point of the run.
    EXPECT_FALSE(ParseTextBackend("").has_value());
    EXPECT_FALSE(ParseTextBackend("sdf").has_value());
    EXPECT_FALSE(ParseTextBackend(" slug ").has_value());
    EXPECT_FALSE(ParseTextBackend("msdf2").has_value());
}

TEST(TextDemoConfigTest, BackendNamesRoundTrip) {
    // Whatever ParseTextBackend accepts must be what TextBackendName prints, so
    // the two spellings cannot drift apart.
    for (const TextBackend backend : {TextBackend::Msdf, TextBackend::Slug}) {
        const auto parsed = ParseTextBackend(TextBackendName(backend));
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(*parsed, backend);
    }
}

TEST(TextDemoConfigTest, SelectionReachesRendererConfig) {
    for (const TextBackend backend : {TextBackend::Msdf, TextBackend::Slug}) {
        const RendererConfig config = MakeRendererConfig(backend);
        EXPECT_EQ(config.text_backend, backend);
        // The screen-space overlay registration is the example's own mapping
        // output: both backends must still request it.
        EXPECT_TRUE(config.enable_text);
    }

    // The example maps the engine's default straight through, so a no-flag run
    // gets whatever backend the engine documents as its default. The engine-
    // owned value of kDefaultTextBackend is asserted in the engine's
    // text-system tests; only the pass-through is the example's business.
    EXPECT_EQ(MakeRendererConfig(kDefaultTextBackend).text_backend, kDefaultTextBackend);
}

// The pass routes a queue call for the wrong family to a no-op, so the example's
// selection has to reach the pass's own Backend(). Constructing the pass with a
// null bootstrap touches no device: the constructor only queries frames-in-flight
// through the bootstrap and creates no buffer without one.
TEST(TextDemoConfigTest, ConfiguredBackendMatchesPassRouting) {
    for (const TextBackend backend : {TextBackend::Msdf, TextBackend::Slug}) {
        const TextBackend configured = MakeRendererConfig(backend).text_backend;
        WorldTextPass pass(nullptr, configured, 0, 0);
        EXPECT_EQ(pass.Backend(), backend);
        EXPECT_EQ(pass.AcceptsMsdfRuns(), backend == TextBackend::Msdf);
        EXPECT_EQ(pass.AcceptsSlugRuns(), backend == TextBackend::Slug);
    }
}

} // namespace
