// Shared hb-gpu (Slug) helpers: the draw encoder's shader sources ported to
// Slang and adapted to this engine in exactly two places.
//
// The source of truth is HarfBuzz's own `hb_gpu_shader_source()` /
// `hb_gpu_draw_shader_source()` HLSL strings (the vertex half is the shared
// `hb_gpu_dilate`, the fragment half is the shared Slug coverage math plus the
// three-line `hb_gpu_draw` wrapper). This file is a near-verbatim carry-over:
// the coverage math, the band decoder, the curve solver and the root-code table
// are copied unchanged, because they are the Slug algorithm and "improving" any
// of them changes the rendered glyphs. A device-free provenance test pins a hash
// of those upstream strings and fails with a re-port instruction when a
// HarfBuzz bump changes them.
//
// Exactly two things are adapted, both called out inline below:
//
//   1. The atlas accessor. `hb_gpu_fetch` is the ONLY place the blob buffer is
//      read, and upstream declares it as `StructuredBuffer<int4>` (16-byte
//      stride). The encoded blob is a sequence of 8-byte units (four signed
//      16-bit values), so that declaration mis-indexes every blob: the buffer is
//      declared as 8-byte `StructuredBuffer<int2>` elements and the four halves
//      are sign-extended into the `int4` the algorithm expects. The extension
//      uses 32-bit shifts only, so it does not need 16-bit storage or arithmetic
//      capability, and it is checked against real blob bytes by the GPU suite.
//
//   2. The matrix convention. `hb_gpu_dilate` upstream is column-vector
//      (`mul (m, float4 (position, ...))`); every engine shader is row-vector
//      (`mul (float4 (p, 1), m)`). The two multiplies are expressed in the
//      engine's convention so the shader is internally consistent with the
//      matrix the push constant carries. A silent transpose here only mis-sizes
//      the half-pixel edge dilation, which is invisible until text looks subtly
//      aliased.
//
// Object space for this port is the blob's own em space: the quad's position and
// texcoord are the same design-unit coordinates, so the em-to-object transform is
// the identity and its inverse (the `jac` argument) is the identity too. The
// model matrix passed in `m` carries the em-to-clip transform, which is where
// the world placement and the requested size enter.

// Set 5 binding 1: the blob storage buffer, 8-byte elements. Declared for the
// fragment stage, which is the only stage that reads it. The caller must bind
// GpuTextBlobBuffer::Buffer() here.
[[vk::binding(1, 5)]]
StructuredBuffer<int2, CDataLayout> hb_gpu_atlas;

// ── Shared fragment helpers (verbatim except hb_gpu_fetch) ─────────────────

#ifndef HB_GPU_UNITS_PER_EM
#define HB_GPU_UNITS_PER_EM 4
#endif

#define HB_GPU_INV_UNITS (1.0 / (float) HB_GPU_UNITS_PER_EM)

// Adaptation 1: upstream is `return hb_gpu_atlas[offset];` over a 16-byte
// `int4`. The blob's unit is 8 bytes, so element `offset` here is the same unit
// upstream meant; each unit holds four signed 16-bit values packed into two
// 32-bit words, and they are sign-extended with arithmetic shifts only.
int4 hb_gpu_fetch (int offset)
{
  int2 raw = hb_gpu_atlas[offset];
  return int4 ((raw.x << 16) >> 16, raw.x >> 16,
               (raw.y << 16) >> 16, raw.y >> 16);
}

uint _hb_gpu_calc_root_code (float y1, float y2, float y3)
{
  uint i1 = asuint (y1) >> 31u;
  uint i2 = asuint (y2) >> 30u;
  uint i3 = asuint (y3) >> 29u;

  uint shift = (i2 & 2u) | (i1 & ~2u);
  shift = (i3 & 4u) | (shift & ~4u);

  return (0x2E74u >> shift) & 0x0101u;
}

float2 _hb_gpu_solve_horiz_poly (float2 a, float2 b, float2 p1)
{
  float ra = 1.0 / a.y;
  float rb = 0.5 / b.y;

  float d = sqrt (max (b.y * b.y - a.y * p1.y, 0.0));
  float t1 = (b.y - d) * ra;
  float t2 = (b.y + d) * ra;

  if (a.y == 0.0)
    t1 = t2 = p1.y * rb;

  return float2 ((a.x * t1 - b.x * 2.0) * t1 + p1.x,
                 (a.x * t2 - b.x * 2.0) * t2 + p1.x);
}

float2 _hb_gpu_solve_vert_poly (float2 a, float2 b, float2 p1)
{
  float ra = 1.0 / a.x;
  float rb = 0.5 / b.x;

  float d = sqrt (max (b.x * b.x - a.x * p1.x, 0.0));
  float t1 = (b.x - d) * ra;
  float t2 = (b.x + d) * ra;

  if (a.x == 0.0)
    t1 = t2 = p1.x * rb;

  return float2 ((a.y * t1 - b.y * 2.0) * t1 + p1.y,
                 (a.y * t2 - b.y * 2.0) * t2 + p1.y);
}

float _hb_gpu_calc_coverage (float xcov, float ycov, float xwgt, float ywgt)
{
  float coverage = max (abs (xcov * xwgt + ycov * ywgt) /
                        max (xwgt + ywgt, 1.0 / 65536.0),
                        min (abs (xcov), abs (ycov)));

  return clamp (coverage, 0.0, 1.0);
}

struct _hb_gpu_glyph_info
{
  int glyphLoc;
  int bandBase;
  int2 bandIndex;
  int numHBands;
  int numVBands;
  float2 scale;
};

_hb_gpu_glyph_info _hb_gpu_decode_glyph (float2 renderCoord, uint glyphLoc_)
{
  _hb_gpu_glyph_info gi;
  gi.glyphLoc = (int) glyphLoc_;

  int4 header0 = hb_gpu_fetch (gi.glyphLoc);
  int4 header1 = hb_gpu_fetch (gi.glyphLoc + 1);
  float4 ext = (float4) header0 * HB_GPU_INV_UNITS;
  gi.numHBands = header1.r;
  gi.numVBands = header1.g;
  gi.scale = float2 ((float) header1.b, (float) header1.a);

  float2 extSize = ext.zw - ext.xy;
  float2 bandScale = float2 ((float) gi.numVBands, (float) gi.numHBands) / max (extSize, float2 (1.0 / 65536.0, 1.0 / 65536.0));
  float2 bandOffset = -ext.xy * bandScale;

  gi.bandIndex = clamp ((int2) (renderCoord * bandScale + bandOffset),
                        int2 (0, 0),
                        int2 (gi.numVBands - 1, gi.numHBands - 1));

  gi.bandBase = gi.glyphLoc + 2;
  return gi;
}

/* Return pixels per em at this fragment.
 *
 * renderCoord:  em-space sample position
 * glyphLoc:     texel offset of glyph blob in atlas
 */
float hb_gpu_ppem (float2 renderCoord, uint glyphLoc_)
{
  _hb_gpu_glyph_info gi = _hb_gpu_decode_glyph (renderCoord, glyphLoc_);
  float2 emsPerPixel = fwidth (renderCoord);
  return min (gi.scale.x, gi.scale.y) /
	 max (emsPerPixel.x, emsPerPixel.y);
}

int2 _hb_gpu_curve_counts (float2 renderCoord, uint glyphLoc_)
{
  _hb_gpu_glyph_info gi = _hb_gpu_decode_glyph (renderCoord, glyphLoc_);
  int hCount = hb_gpu_fetch (gi.bandBase + gi.bandIndex.y).r;
  int vCount = hb_gpu_fetch (gi.bandBase + gi.numHBands + gi.bandIndex.x).r;
  return int2 (hCount, vCount);
}

/* Single-sample coverage in [0, 1]. */
float _hb_gpu_slug_single (float2 renderCoord, float2 pixelsPerEm, uint glyphLoc_)
{

  _hb_gpu_glyph_info gi = _hb_gpu_decode_glyph (renderCoord, glyphLoc_);
  int glyphLoc = gi.glyphLoc;
  int bandBase = gi.bandBase;
  int numHBands = gi.numHBands;

  float xcov = 0.0;
  float xwgt = 0.0;

  int4 hbandData = hb_gpu_fetch (bandBase + gi.bandIndex.y);
  int hCurveCount = hbandData.r;
  float hSplit = (float) hbandData.a * HB_GPU_INV_UNITS;
  bool hLeftRay = (renderCoord.x < hSplit);
  int hDataOffset = (hLeftRay ? hbandData.b : hbandData.g) + 32768;

  for (int ci = 0; ci < hCurveCount; ci++)
  {
    int curveOffset = hb_gpu_fetch (glyphLoc + hDataOffset + ci).r + 32768;

    int4 raw12 = hb_gpu_fetch (glyphLoc + curveOffset);
    int4 raw3 = hb_gpu_fetch (glyphLoc + curveOffset + 1);

    float4 q12 = (float4) raw12 * HB_GPU_INV_UNITS;
    float2 q3 = (float2) raw3.rg * HB_GPU_INV_UNITS;

    float4 p12 = q12 - float4 (renderCoord, renderCoord);
    float2 p3 = q3 - renderCoord;

    if (hLeftRay) {
      if (min (min (p12.x, p12.z), p3.x) * pixelsPerEm.x > 0.5) break;
    } else {
      if (max (max (p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) break;
    }

    uint code = _hb_gpu_calc_root_code (p12.y, p12.w, p3.y);
    if (code != 0u)
    {
      float2 a = q12.xy - q12.zw * 2.0 + q3;
      float2 b = q12.xy - q12.zw;
      float2 r = _hb_gpu_solve_horiz_poly (a, b, p12.xy) * pixelsPerEm.x;
      float2 cov = hLeftRay ? clamp (float2 (0.5, 0.5) - r, 0.0, 1.0)
                            : clamp (r + float2 (0.5, 0.5), 0.0, 1.0);

      if ((code & 1u) != 0u)
      {
        xcov += cov.x;
        xwgt = max (xwgt, clamp (1.0 - abs (r.x) * 2.0, 0.0, 1.0));
      }

      if (code > 1u)
      {
        xcov -= cov.y;
        xwgt = max (xwgt, clamp (1.0 - abs (r.y) * 2.0, 0.0, 1.0));
      }
    }
  }

  // Crossings over a closed contour sum to zero, so a leftward ray
  // returns the negative of what a rightward ray would; flip it back
  // so that xcov and ycov keep a common sign convention. (14.5.1)
  if (hLeftRay)
    xcov = -xcov;

  float ycov = 0.0;
  float ywgt = 0.0;

  int4 vbandData = hb_gpu_fetch (bandBase + numHBands + gi.bandIndex.x);
  int vCurveCount = vbandData.r;
  float vSplit = (float) vbandData.a * HB_GPU_INV_UNITS;
  bool vLeftRay = (renderCoord.y < vSplit);
  int vDataOffset = (vLeftRay ? vbandData.b : vbandData.g) + 32768;

  for (int ci = 0; ci < vCurveCount; ci++)
  {
    int curveOffset = hb_gpu_fetch (glyphLoc + vDataOffset + ci).r + 32768;

    int4 raw12 = hb_gpu_fetch (glyphLoc + curveOffset);
    int4 raw3 = hb_gpu_fetch (glyphLoc + curveOffset + 1);

    float4 q12 = (float4) raw12 * HB_GPU_INV_UNITS;
    float2 q3 = (float2) raw3.rg * HB_GPU_INV_UNITS;

    float4 p12 = q12 - float4 (renderCoord, renderCoord);
    float2 p3 = q3 - renderCoord;

    if (vLeftRay) {
      if (min (min (p12.y, p12.w), p3.y) * pixelsPerEm.y > 0.5) break;
    } else {
      if (max (max (p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) break;
    }

    uint code = _hb_gpu_calc_root_code (p12.x, p12.z, p3.x);
    if (code != 0u)
    {
      float2 a = q12.xy - q12.zw * 2.0 + q3;
      float2 b = q12.xy - q12.zw;
      float2 r = _hb_gpu_solve_vert_poly (a, b, p12.xy) * pixelsPerEm.y;
      float2 cov = vLeftRay ? clamp (float2 (0.5, 0.5) - r, 0.0, 1.0)
                            : clamp (r + float2 (0.5, 0.5), 0.0, 1.0);

      if ((code & 1u) != 0u)
      {
        ycov -= cov.x;
        ywgt = max (ywgt, clamp (1.0 - abs (r.x) * 2.0, 0.0, 1.0));
      }

      if (code > 1u)
      {
        ycov += cov.y;
        ywgt = max (ywgt, clamp (1.0 - abs (r.y) * 2.0, 0.0, 1.0));
      }
    }
  }

  // Ditto, for the vertical ray. (14.5.1)
  if (vLeftRay)
    ycov = -ycov;

  return _hb_gpu_calc_coverage (xcov, ycov, xwgt, ywgt);
}

/* Return coverage in [0, 1].
 *
 * renderCoord:  em-space sample position
 * glyphLoc:     texel offset of glyph blob in atlas
 */
/* The MSAA-aware implementation.  Caller supplies pixelsPerEm so
 * this function can be invoked from non-uniform control flow (for
 * example from a paint op-stream branch). */
float _hb_gpu_slug (float2 renderCoord, float2 pixelsPerEm, uint glyphLoc_)
{
  float c = _hb_gpu_slug_single (renderCoord, pixelsPerEm, glyphLoc_);

#ifndef HB_GPU_NO_MSAA
  float ppem = hb_gpu_ppem (renderCoord, glyphLoc_);

  if (ppem < 16.0)
  {
    float2 emsPerPixel = 1.0 / pixelsPerEm;
    float2 d = emsPerPixel * (1.0 / 3.0);
    float msaa = 0.25 *
      (_hb_gpu_slug_single (renderCoord + float2 (-d.x, -d.y), pixelsPerEm, glyphLoc_) +
       _hb_gpu_slug_single (renderCoord + float2 ( d.x, -d.y), pixelsPerEm, glyphLoc_) +
       _hb_gpu_slug_single (renderCoord + float2 (-d.x,  d.y), pixelsPerEm, glyphLoc_) +
       _hb_gpu_slug_single (renderCoord + float2 ( d.x,  d.y), pixelsPerEm, glyphLoc_));

    c = lerp (c, msaa, smoothstep (16.0, 8.0, ppem));
  }
#endif

  return c;
}

/* Stem darkening for small sizes.
 *
 * coverage:    output of hb_gpu_draw
 * brightness:  foreground brightness in [0, 1]
 * ppem:        pixels per em at this fragment
 */
float hb_gpu_stem_darken (float coverage, float brightness, float ppem)
{
  return pow (coverage,
	      lerp (pow (2.0, brightness - 0.5), 1.0,
		    smoothstep (8.0, 48.0, ppem)));
}

/* Draw-renderer fragment entry (HLSL) carry-over.  Heavy lifting (Slug
 * coverage, MSAA, ppem, stem darkening) lives above. */
float hb_gpu_draw (float2 renderCoord, uint glyphLoc_)
{
  float2 pixelsPerEm = 1.0 / fwidth (renderCoord);
  return _hb_gpu_slug (renderCoord, pixelsPerEm, glyphLoc_);
}

// ── Shared vertex helper: hb_gpu_dilate (adaptation 2) ─────────────────────

/* Dilate a glyph vertex by half a pixel on screen.
 *
 * position:  object-space vertex position (modified in place)
 * texcoord:  em-space sample coordinates (modified in place)
 * normal:    object-space outward normal at this vertex
 * jac:       inverse of the 2x2 linear part of the em-to-object transform
 * m:         model-view-projection matrix
 * viewport:  viewport size in pixels
 */
void hb_gpu_dilate (inout float2 position, inout float2 texcoord,
                    float2 normal, float4 jac,
                    float4x4 m, float2 viewport)
{
  float2 n = normalize (normal);

  // Adaptation 2: the engine's row-vector convention. Upstream is
  // `mul (m, float4 (position, 0.0, 1.0))`; the matrix must be applied on the
  // right of the row vector everywhere this port uses it, including the
  // matching `mul (float4 (pos, 0.0, 1.0), m)` in the vertex entry point.
  float4 clipPos = mul (float4 (position, 0.0, 1.0), m);
  float4 clipN   = mul (float4 (n, 0.0, 0.0), m);

  float s = clipPos.w;
  float t = clipN.w;

  float u = (s * clipN.x - t * clipPos.x) * viewport.x;
  float v = (s * clipN.y - t * clipPos.y) * viewport.y;

  float s2 = s * s;
  float st = s * t;
  float uv = u * u + v * v;

  float denom = uv - st * st;
  float d = abs (denom) > 1.0 / 16777216.0
          ? s2 * (st + sqrt (uv)) / denom
          : 0.0;

  float2 dPos = d * normal;
  position += dPos;
  texcoord += float2 (dot (dPos, jac.xy), dot (dPos, jac.zw));
}
