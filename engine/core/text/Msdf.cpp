module;

#include <hb.h>
#include <msdfgen.h>

module VulkanEngine.Text.Msdf;

import std;
import std.compat;

import VulkanEngine.Text.Atlas;
import VulkanEngine.Text.Font;

namespace VulkanEngine::Text {

namespace {

// boost-style hash_combine, same as the shaping and glyph caches'. The hash only
// chooses a bucket; equality decides whether two keys are the same entry.
[[nodiscard]] std::size_t MixHash(std::size_t seed, std::size_t value) noexcept {
    return seed ^ (value + static_cast<std::size_t>(0x9E3779B97F4A7C15ULL) + (seed << 6U) +
                   (seed >> 2U));
}

// A field wider or taller than this is not a glyph; it is a misrequest or a
// corrupt face. Bounding it here keeps the generator from allocating an
// unbounded bitmap before the atlas ever gets a chance to reject the slot.
constexpr std::uint32_t kMaxFieldExtent = 4096;

[[nodiscard]] bool SamePoint(const msdfgen::Point2& a, const msdfgen::Point2& b) noexcept {
    return a.x == b.x && a.y == b.y;
}

// The mutable state the HarfBuzz callbacks fill. One per Draw() call, and there
// is no shared state between them, which is why the immutable callback table can
// be shared across threads.
struct DrawBuilder {
    msdfgen::Shape* shape = nullptr;
    // Optional parallel description of the same contours, for Describe().
    GlyphOutline* outline = nullptr;
    msdfgen::Contour* contour = nullptr;
    msdfgen::Point2 start{};
    msdfgen::Point2 current{};
    bool open = false;
    // Draw-work budget (HarfBuzz 14.5): the glyf/ot draw path spends from the
    // counter this struct owns, bounding pathological fonts on the mesh
    // thread. An exhausted budget surfaces as draw_glyph_or_fail returning
    // false, which Draw() already treats as "no drawable outline".
    int64_t budget = 0;
    int64_t budget_remaining = 0;
};

void RecordEdge(const DrawBuilder& builder, OutlineEdgeKind kind) {
    if (builder.outline != nullptr && !builder.outline->contours.empty()) {
        builder.outline->contours.back().edges.push_back(kind);
    }
}

// Finishes the contour the callbacks are building. A close_path closes it now; a
// move_to, or the end of the draw, closes whatever is still open. A contour that
// has not returned to its start point gets the closing edge HarfBuzz's implicit
// closing implies, so msdfgen sees the same closed contour the font means.
void CloseContour(DrawBuilder& builder, bool mark_closed) {
    if (builder.open && builder.contour != nullptr && !SamePoint(builder.current, builder.start)) {
        builder.contour->addEdge(
            msdfgen::EdgeHolder(builder.current, builder.start));
        RecordEdge(builder, OutlineEdgeKind::Line);
    }
    if (mark_closed && builder.outline != nullptr && !builder.outline->contours.empty()) {
        builder.outline->contours.back().closed = true;
    }
    builder.open = false;
    builder.contour = nullptr;
}

void MoveTo(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, hb_draw_state_t* /*st*/, float to_x,
            float to_y, void* /*user_data*/) {
    auto& builder = *static_cast<DrawBuilder*>(draw_data);
    CloseContour(builder, false);
    builder.shape->contours.emplace_back();
    builder.contour = &builder.shape->contours.back();
    builder.start = msdfgen::Point2(to_x, to_y);
    builder.current = builder.start;
    builder.open = true;
    if (builder.outline != nullptr) {
        builder.outline->contours.emplace_back();
    }
}

void LineTo(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, hb_draw_state_t* /*st*/, float to_x,
            float to_y, void* /*user_data*/) {
    auto& builder = *static_cast<DrawBuilder*>(draw_data);
    if (!builder.open) {
        return;
    }
    const msdfgen::Point2 to(to_x, to_y);
    builder.contour->addEdge(msdfgen::EdgeHolder(builder.current, to));
    RecordEdge(builder, OutlineEdgeKind::Line);
    builder.current = to;
}

void QuadraticTo(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, hb_draw_state_t* /*st*/,
                 float control_x, float control_y, float to_x, float to_y, void* /*user_data*/) {
    auto& builder = *static_cast<DrawBuilder*>(draw_data);
    if (!builder.open) {
        return;
    }
    const msdfgen::Point2 control(control_x, control_y);
    const msdfgen::Point2 to(to_x, to_y);
    builder.contour->addEdge(msdfgen::EdgeHolder(builder.current, control, to));
    RecordEdge(builder, OutlineEdgeKind::Quadratic);
    builder.current = to;
}

void CubicTo(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, hb_draw_state_t* /*st*/,
             float control1_x, float control1_y, float control2_x, float control2_y, float to_x,
             float to_y, void* /*user_data*/) {
    auto& builder = *static_cast<DrawBuilder*>(draw_data);
    if (!builder.open) {
        return;
    }
    const msdfgen::Point2 control1(control1_x, control1_y);
    const msdfgen::Point2 control2(control2_x, control2_y);
    const msdfgen::Point2 to(to_x, to_y);
    builder.contour->addEdge(msdfgen::EdgeHolder(builder.current, control1, control2, to));
    RecordEdge(builder, OutlineEdgeKind::Cubic);
    builder.current = to;
}

void ClosePath(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, hb_draw_state_t* /*st*/,
               void* /*user_data*/) {
    CloseContour(*static_cast<DrawBuilder*>(draw_data), true);
}

// ── Budget plumbing (HarfBuzz 14.5 hb-draw budget hooks) ────────────────────
// HarfBuzz's own outline extraction spends from the live budget counter; these
// three functions only expose DrawBuilder's policy and counter to it.

hb_bool_t SetBudget(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, int64_t budget,
                    void* /*user_data*/) {
    auto& builder = *static_cast<DrawBuilder*>(draw_data);
    // Mirror upstream: negative policies other than DEFAULT clamp to zero.
    builder.budget = (budget < 0 && budget != HB_BUDGET_DEFAULT) ? 0 : budget;
    builder.budget_remaining = builder.budget; // set recharges the live budget
    return true;
}

int64_t GetBudget(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, void* /*user_data*/) {
    return static_cast<DrawBuilder*>(draw_data)->budget;
}

int64_t* GetBudgetRemaining(hb_draw_funcs_t* /*dfuncs*/, void* draw_data, void* /*user_data*/) {
    return &static_cast<DrawBuilder*>(draw_data)->budget_remaining;
}

} // namespace

std::size_t GlyphOutline::EdgeCount() const noexcept {
    std::size_t total = 0;
    for (const OutlineContour& contour : contours) {
        total += contour.edges.size();
    }
    return total;
}

struct HarfBuzzOutlineAdapter::Impl {
    hb_draw_funcs_t* funcs = nullptr;

    // Generous per-glyph draw budget resolved from HB_BUDGET_DEFAULT policy:
    // a real glyph has at most a few thousand points; a million draw units
    // catches pathological fonts without ever limiting realistic ones.
    static constexpr int64_t kDrawBudgetPerGlyph = 1'000'000;

    Impl() = default;
    ~Impl() {
        if (funcs != nullptr) {
            hb_draw_funcs_destroy(funcs);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Fills `shape` (and, when non-null, `outline`) from the glyph's outline.
    // Returns false when HarfBuzz has no drawable outline for the glyph or the
    // callbacks produced nothing.
    [[nodiscard]] bool Draw(const FontFace& face, std::uint32_t glyph_id,
                            msdfgen::Shape& shape, GlyphOutline* outline) const {
        if (funcs == nullptr || face.HarfBuzzFont() == nullptr) {
            return false;
        }
        DrawBuilder builder;
        builder.shape = &shape;
        builder.outline = outline;
        // HB_BUDGET_DEFAULT resolves here to a generous per-glyph draw count;
        // HarfBuzz spends from the counter as it extracts the outline, so a
        // pathological font (millions of points) fails the draw - the glyph
        // is skipped - instead of running unbounded worker work.
        builder.budget = kDrawBudgetPerGlyph;
        builder.budget_remaining = kDrawBudgetPerGlyph;
        const hb_bool_t drawn =
            hb_font_draw_glyph_or_fail(face.HarfBuzzFont(), glyph_id, funcs, &builder);
        CloseContour(builder, false);
        return drawn != 0 && !shape.contours.empty();
    }
};

namespace {

// Generates one glyph's field. `padding` is the requested range rounded up, in
// field texels; the field is the ink box inset into that gutter on every side, so
// the outline is at least `range` texels away from every edge.
[[nodiscard]] std::shared_ptr<MsdfGlyph> GenerateField(const HarfBuzzOutlineAdapter::Impl& adapter,
                                                       const FontFace& face,
                                                       std::uint32_t glyph_id, float field_size,
                                                       double range) {
    auto glyph = std::make_shared<MsdfGlyph>();
    glyph->glyph_id = glyph_id;
    glyph->range = static_cast<float>(range);

    // The ink box comes from the same HarfBuzz extents GlyphAtSize reports, so
    // the field and the layout side agree on where a glyph is.
    const GlyphMetrics metrics = face.GlyphAtSize(glyph_id, field_size);
    glyph->advance_x = metrics.advance_x;
    if (!(metrics.width > 0.0f) || !(metrics.height > 0.0f)) {
        // A space, or a glyph with no outline: a zero-sized field and no pixels.
        return glyph;
    }
    const float scale = face.ScaleForSize(field_size);
    if (!(scale > 0.0f)) {
        return nullptr;
    }

    const std::uint32_t padding = static_cast<std::uint32_t>(std::ceil(range));
    const std::uint64_t width =
        static_cast<std::uint64_t>(std::ceil(static_cast<double>(metrics.width))) + 2ULL * padding;
    const std::uint64_t height =
        static_cast<std::uint64_t>(std::ceil(static_cast<double>(metrics.height))) + 2ULL * padding;
    if (width == 0 || height == 0 || width > kMaxFieldExtent || height > kMaxFieldExtent) {
        return nullptr;
    }
    glyph->width = static_cast<std::uint32_t>(width);
    glyph->height = static_cast<std::uint32_t>(height);
    // Y-down placement, the same convention RasterGlyph uses; the range padding
    // pushes the field's top-left up and left of the ink box.
    glyph->left = metrics.bearing_x - static_cast<float>(padding);
    glyph->top = -metrics.bearing_y - static_cast<float>(padding);

    msdfgen::Shape shape;
    // Design units are y-up, which is msdfgen's shape space: no transform and no
    // axis inversion are needed.
    shape.inverseYAxis = false;
    if (!adapter.Draw(face, glyph_id, shape, nullptr)) {
        return nullptr;
    }
    // A contour that is not oriented inverts the fill rule: msdfgen would report
    // the exterior as the interior. TrueType/OpenType outlines carry no explicit
    // orientation guarantee at the draw-callback level, so orient them.
    shape.orientContours();
    msdfgen::edgeColoringSimple(shape, 3.0);

    // Project design units into the field: x maps design [bearing_x, +width] onto
    // [padding, padding + width], and y is negated so the y-up outline lands in a
    // top-down bitmap with row 0 at the glyph's top.
    const double s = static_cast<double>(scale);
    const msdfgen::Projection projection(
        msdfgen::Vector2(s, -s),
        msdfgen::Vector2(
            (static_cast<double>(padding) - static_cast<double>(metrics.bearing_x)) / s,
            -(static_cast<double>(padding) + static_cast<double>(metrics.bearing_y)) / s));

    msdfgen::Bitmap<float, 4> field(static_cast<int>(glyph->width),
                                    static_cast<int>(glyph->height));
    // The range is a distance in field texels; msdfgen measures distance in shape
    // units, so it is divided by the projection scale.
    msdfgen::generateMTSDF(field, shape, projection, msdfgen::Range(range / s),
                           msdfgen::MSDFGeneratorConfig());

    glyph->pixels.resize(static_cast<std::size_t>(glyph->width) * glyph->height *
                         MsdfGenerator::kChannels);
    for (std::uint32_t y = 0; y < glyph->height; ++y) {
        for (std::uint32_t x = 0; x < glyph->width; ++x) {
            const float* texel = field(static_cast<int>(x), static_cast<int>(y));
            const std::size_t base =
                (static_cast<std::size_t>(y) * glyph->width + x) * MsdfGenerator::kChannels;
            for (std::uint32_t channel = 0; channel < MsdfGenerator::kChannels; ++channel) {
                glyph->pixels[base + channel] = msdfgen::pixelFloatToByte(texel[channel]);
            }
        }
    }
    return glyph;
}

}

HarfBuzzOutlineAdapter::HarfBuzzOutlineAdapter() : impl_(std::make_unique<Impl>()) {
    impl_->funcs = hb_draw_funcs_create();
    if (impl_->funcs == nullptr) {
        return;
    }
    hb_draw_funcs_set_move_to_func(impl_->funcs, &MoveTo, nullptr, nullptr);
    hb_draw_funcs_set_line_to_func(impl_->funcs, &LineTo, nullptr, nullptr);
    hb_draw_funcs_set_quadratic_to_func(impl_->funcs, &QuadraticTo, nullptr, nullptr);
    hb_draw_funcs_set_cubic_to_func(impl_->funcs, &CubicTo, nullptr, nullptr);
    hb_draw_funcs_set_close_path_func(impl_->funcs, &ClosePath, nullptr, nullptr);
    hb_draw_funcs_set_set_budget_func(impl_->funcs, &SetBudget, nullptr, nullptr);
    hb_draw_funcs_set_get_budget_func(impl_->funcs, &GetBudget, nullptr, nullptr);
    hb_draw_funcs_set_get_budget_remaining_func(impl_->funcs, &GetBudgetRemaining, nullptr, nullptr);
    // Immutable after the setters: HarfBuzz then allows the table to be used from
    // any thread at the same time, which is what makes one adapter shareable.
    hb_draw_funcs_make_immutable(impl_->funcs);
}

HarfBuzzOutlineAdapter::~HarfBuzzOutlineAdapter() = default;

std::optional<GlyphOutline> HarfBuzzOutlineAdapter::Describe(const FontFace& face,
                                                             std::uint32_t glyph_id) const {
    if (!impl_) {
        return std::nullopt;
    }
    msdfgen::Shape shape;
    shape.inverseYAxis = false;
    GlyphOutline outline;
    if (!impl_->Draw(face, glyph_id, shape, &outline)) {
        return std::nullopt;
    }
    return outline;
}

float QuantizeFieldSize(float field_pixel_size) noexcept {
    if (!(field_pixel_size > 0.0f)) {
        // Zero, negative or NaN: "no size" rather than a bogus eighth.
        return 0.0f;
    }
    return std::round(field_pixel_size * 8.0f) / 8.0f;
}

MsdfGenerator::MsdfGenerator(AtlasConfig atlas_config, std::size_t max_cached_glyphs)
    : atlas_(atlas_config), max_entries_(max_cached_glyphs) {}

std::size_t MsdfGenerator::KeyHash::operator()(const Key& key) const noexcept {
    std::size_t hash = std::hash<std::uint64_t>{}(key.face_id);
    hash = MixHash(hash, std::hash<std::uint32_t>{}(key.resource_version));
    hash = MixHash(hash, std::hash<std::uint32_t>{}(key.glyph_id));
    hash = MixHash(hash,
                   std::hash<std::uint32_t>{}(std::bit_cast<std::uint32_t>(key.field_pixel_size)));
    hash = MixHash(hash, std::hash<std::uint64_t>{}(key.range_bits));
    return hash;
}

std::optional<GlyphSlot> MsdfGenerator::Generate(const FontFace& face, std::uint32_t glyph_id,
                                                 const MsdfConfig& config) {
    return GetOrGenerate(face, glyph_id, config).slot;
}

std::shared_ptr<const MsdfGlyph> MsdfGenerator::Get(const FontFace& face, std::uint32_t glyph_id,
                                                    const MsdfConfig& config) {
    return GetOrGenerate(face, glyph_id, config).glyph;
}

std::size_t MsdfGenerator::Size() const {
    const std::scoped_lock lock(mutex_);
    return entries_.size();
}

std::uint64_t MsdfGenerator::HitCount() const {
    const std::scoped_lock lock(mutex_);
    return hits_;
}

void MsdfGenerator::Clear() {
    const std::scoped_lock lock(mutex_);
    for (const Entry& entry : entries_) {
        if (entry.in_atlas) {
            static_cast<void>(atlas_.Erase(entry.atlas_key));
        }
    }
    entries_.clear();
    index_.clear();
}

void MsdfGenerator::Reset() {
    const std::scoped_lock lock(mutex_);
    entries_.clear();
    index_.clear();
    atlas_.Reset();
}

std::shared_ptr<const AtlasBitmap> MsdfGenerator::BitmapForAtlasKey(
    std::uint64_t atlas_key) const {
    const std::scoped_lock lock(mutex_);
    for (const Entry& entry : entries_) {
        if (entry.in_atlas && entry.atlas_key == atlas_key) {
            return entry.bitmap;
        }
    }
    return nullptr;
}

MsdfGenerator::Result MsdfGenerator::GetOrGenerate(const FontFace& face, std::uint32_t glyph_id,
                                                   const MsdfConfig& config) {
    const float quantized = QuantizeFieldSize(config.field_pixel_size);
    if (quantized <= 0.0f) {
        return {};
    }
    const Key key{face.UniqueId(), face.ResourceVersion(), glyph_id, quantized,
                  std::bit_cast<std::uint64_t>(config.range)};

    {
        const std::scoped_lock lock(mutex_);
        const auto found = index_.find(key);
        if (found != index_.end()) {
            ++hits_;
            entries_.splice(entries_.begin(), entries_, found->second);
            return Result{found->second->glyph, found->second->slot};
        }
    }

    // Generate without the lock: it is the expensive part, and holding the mutex
    // across it would serialize the workers a batch generation exists to feed.
    // Two threads racing on one key simply generate it twice and the loser reuses
    // the winner's entry below.
    auto glyph = GenerateField(*adapter_.impl_, face, glyph_id, quantized, config.range);
    if (glyph == nullptr) {
        return {};
    }

    const std::scoped_lock lock(mutex_);
    if (const auto existing = index_.find(key); existing != index_.end()) {
        ++hits_;
        entries_.splice(entries_.begin(), entries_, existing->second);
        return Result{existing->second->glyph, existing->second->slot};
    }

    if (max_entries_ == 0) {
        // Caching disabled: hand back the field but reserve nothing, so the atlas
        // cannot accumulate slots no entry will ever release.
        return Result{glyph, std::nullopt};
    }

    Entry entry;
    entry.key = key;
    entry.glyph = glyph;
    if (!glyph->Empty() && !glyph->pixels.empty()) {
        auto bitmap = std::make_shared<AtlasBitmap>();
        bitmap->width = glyph->width;
        bitmap->height = glyph->height;
        bitmap->bytes = glyph->pixels;
        entry.bitmap = std::move(bitmap);

        // One atlas key per cache entry, handed out monotonically: two different
        // fields can never collide on one rectangle.
        entry.atlas_key = next_atlas_key_++;
        if (atlas_.Insert(entry.atlas_key, glyph->width, glyph->height) != AtlasInsert::TooLarge) {
            entry.slot = atlas_.Find(entry.atlas_key);
            entry.in_atlas = entry.slot.has_value();
        }
    }

    const Result result{entry.glyph, entry.slot};
    entries_.push_front(std::move(entry));
    index_.emplace(entries_.front().key, entries_.begin());
    while (entries_.size() > max_entries_) {
        const Entry& victim = entries_.back();
        if (victim.in_atlas) {
            static_cast<void>(atlas_.Erase(victim.atlas_key));
        }
        index_.erase(victim.key);
        entries_.pop_back();
    }
    return result;
}

} // namespace VulkanEngine::Text
