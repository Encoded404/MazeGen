struct MazeConstants {
    float4 view;   // center_x, center_y, pixels_per_cell, wall_thickness_cells
    float4 screen; // width, height, tier (0/1/2), density block size
    uint4 maze;    // width, height, tiles_x, density mip count
};

[[vk::push_constant]]
MazeConstants params;

[[vk::binding(0, 5)]]
StructuredBuffer<uint> mazeWords;

[[vk::binding(1, 5)]]
StructuredBuffer<float> densityPyramid;

struct PSInput {
    float4 position : SV_Position;
};

static const float4 kBackground = float4(0.02, 0.02, 0.03, 1.0);
static const float4 kFloor = float4(0.07, 0.08, 0.11, 1.0);
static const float4 kWall = float4(0.82, 0.85, 0.90, 1.0);

// Two bits per cell, north in bit 0 and west in bit 1. Tiles are 16x16 cells
// with eight 64-bit words each; the buffer is uploaded as 32-bit words.
bool MazeBit(uint x, uint y, uint offset) {
    const uint width = params.maze.x;
    const uint height = params.maze.y;
    if (x > width || y > height) {
        return false;
    }
    if (offset == 0u) {
        if (x >= width || y > height) {
            return false;
        }
    } else {
        if (x > width || y >= height) {
            return false;
        }
    }

    const uint tile_x = x >> 4u;
    const uint tile_y = y >> 4u;
    const uint local = (y & 15u) * 16u + (x & 15u);
    const uint word64 = (tile_y * params.maze.z + tile_x) * 8u + (local >> 5u);
    const uint shift = 2u * (local & 31u) + offset;
    const uint low = mazeWords[word64 * 2u];
    const uint high = mazeWords[word64 * 2u + 1u];
    return (shift < 32u ? ((low >> shift) & 1u) : ((high >> (shift - 32u)) & 1u)) != 0u;
}

bool NorthWall(uint x, uint y) {
    return MazeBit(x, y, 0u);
}

bool WestWall(uint x, uint y) {
    return MazeBit(x, y, 1u);
}

// Exact tier: binary pixel-locked grid. The half-open comparison is the
// tie-break that keeps an integer P from producing two (or zero) wall pixels
// when a cell boundary lands exactly on a pixel edge.
float4 ExactMaze(float2 pixel) {
    const float pitch = params.view.z;
    const float thickness = params.view.w;
    const float2 center = params.view.xy;
    const float2 screen = params.screen.xy;
    const int width = int(params.maze.x);
    const int height = int(params.maze.y);

    const float2 origin = round(center * pitch - 0.5 * screen);
    const float2 u = (pixel + origin) / pitch;
    const float2 cell_f = floor(u);
    const float2 f = u - cell_f;
    const int cx = int(cell_f.x);
    const int cy = int(cell_f.y);

    if (cx < 0 || cy < 0 || cx >= width || cy >= height) {
        return kBackground;
    }

    const int wall_px = clamp(int(round(thickness * pitch)), 1, max(1, int(floor(pitch)) - 1));
    const float half_px = 0.5 * float(wall_px);

    const bool left = f.x * pitch <= half_px;
    const bool right = (1.0 - f.x) * pitch < half_px;
    const bool top = f.y * pitch <= half_px;
    const bool bottom = (1.0 - f.y) * pitch < half_px;

    bool wall = false;

    if (left || right) {
        const int bx = left ? cx : cx + 1;
        wall = wall || (cy >= 0 && cy < height && WestWall(uint(bx), uint(cy)));
        if (top && cy - 1 >= 0 && cy - 1 < height) {
            wall = wall || WestWall(uint(bx), uint(cy - 1));
        }
        if (bottom && cy + 1 >= 0 && cy + 1 < height) {
            wall = wall || WestWall(uint(bx), uint(cy + 1));
        }
    }

    if (top || bottom) {
        const int by = top ? cy : cy + 1;
        wall = wall || (cx >= 0 && cx < width && NorthWall(uint(cx), uint(by)));
        if (left && cx - 1 >= 0 && cx - 1 < width) {
            wall = wall || NorthWall(uint(cx - 1), uint(by));
        }
        if (right && cx + 1 >= 0 && cx + 1 < width) {
            wall = wall || NorthWall(uint(cx + 1), uint(by));
        }
    }

    return wall ? kWall : kFloor;
}

// Analytic tier: box-filtered wall rectangles. The effective wall width tapers
// from one pixel at P=2 to the physical thickness by P=1, so the lower band
// does not saturate into a solid wall field.
float4 AnalyticMaze(float2 pixel) {
    const float pitch = params.view.z;
    const float thickness = params.view.w;
    const float2 center = params.view.xy;
    const float2 screen = params.screen.xy;
    const uint width = params.maze.x;
    const uint height = params.maze.y;

    const float s = 1.0 / pitch;
    const float2 u = center + (pixel - 0.5 * screen) / pitch;
    // Outside the maze is empty background, not floor. The exact and density
    // tiers already bound their sampling; without this check the floor color
    // would leak into the region revealed by overscroll panning.
    if (u.x < 0.0 || u.y < 0.0 || u.x >= float(width) || u.y >= float(height)) {
        return kBackground;
    }
    const float2 lo = u - 0.5 * s;
    const float2 hi = u + 0.5 * s;

    const float screen_wall_px = max(thickness * pitch, clamp(pitch - 1.0, 0.0, 1.0));
    const float t_eff = screen_wall_px / pitch;
    const float half = 0.5 * t_eff;

    float area = 0.0;

    // Vertical walls: boundary x = ix, cell row [iy, iy + 1].
    const int vx0 = int(floor(lo.x - half));
    const int vx1 = int(ceil(hi.x + half));
    const int vy0 = int(floor(lo.y)) - 1;
    const int vy1 = int(ceil(hi.y));
    for (int ix = vx0; ix <= vx1; ++ix) {
        if (ix < 0 || ix > int(width)) {
            continue;
        }
        for (int iy = vy0; iy <= vy1; ++iy) {
            if (iy < 0 || iy >= int(height) || !WestWall(uint(ix), uint(iy))) {
                continue;
            }
            const float ax0 = max(lo.x, float(ix) - half);
            const float ax1 = min(hi.x, float(ix) + half);
            const float ay0 = max(lo.y, float(iy));
            const float ay1 = min(hi.y, float(iy + 1));
            if (ax1 > ax0 && ay1 > ay0) {
                area += (ax1 - ax0) * (ay1 - ay0);
            }
        }
    }

    // Horizontal walls: boundary y = iy, cell column [ix, ix + 1].
    const int hy0 = int(floor(lo.y - half));
    const int hy1 = int(ceil(hi.y + half));
    const int hx0 = int(floor(lo.x)) - 1;
    const int hx1 = int(ceil(hi.x));
    for (int iy = hy0; iy <= hy1; ++iy) {
        if (iy < 0 || iy > int(height)) {
            continue;
        }
        for (int ix = hx0; ix <= hx1; ++ix) {
            if (ix < 0 || ix >= int(width) || !NorthWall(uint(ix), uint(iy))) {
                continue;
            }
            const float ax0 = max(lo.x, float(ix));
            const float ax1 = min(hi.x, float(ix + 1));
            const float ay0 = max(lo.y, float(iy) - half);
            const float ay1 = min(hi.y, float(iy) + half);
            if (ax1 > ax0 && ay1 > ay0) {
                area += (ax1 - ax0) * (ay1 - ay0);
            }
        }
    }

    const float coverage = clamp(area / (s * s), 0.0, 1.0);
    return lerp(kFloor, kWall, coverage);
}

float SampleDensityLevel(float2 cell, uint level) {
    uint level_width = (params.maze.x + 3u) / 4u;
    uint level_height = (params.maze.y + 3u) / 4u;
    uint offset = 0u;
    for (uint i = 0u; i < level; ++i) {
        offset += level_width * level_height;
        level_width = (level_width + 1u) / 2u;
        level_height = (level_height + 1u) / 2u;
    }

    const float scale = float(1u << level);
    const float2 p = cell * (0.25 / scale) - 0.5;
    int2 i0 = int2(floor(p));
    const float2 frac = p - floor(p);
    int2 i1 = i0 + int2(1, 1);

    i0.x = clamp(i0.x, 0, int(level_width) - 1);
    i0.y = clamp(i0.y, 0, int(level_height) - 1);
    i1.x = clamp(i1.x, 0, int(level_width) - 1);
    i1.y = clamp(i1.y, 0, int(level_height) - 1);

    const uint row0 = uint(i0.y) * level_width;
    const uint row1 = uint(i1.y) * level_width;
    const float a = densityPyramid[offset + row0 + uint(i0.x)];
    const float b = densityPyramid[offset + row0 + uint(i1.x)];
    const float c = densityPyramid[offset + row1 + uint(i0.x)];
    const float d = densityPyramid[offset + row1 + uint(i1.x)];
    return lerp(lerp(a, b, frac.x), lerp(c, d, frac.x), frac.y);
}

// Density tier: a mipmapped average wall fraction. Physical thickness is used
// here deliberately; a one-pixel floor would cover more than a cell at this
// zoom and turn the whole view into wall colour.
float4 DensityMaze(float2 pixel) {
    const float pitch = params.view.z;
    const float2 center = params.view.xy;
    const float2 screen = params.screen.xy;
    const float2 u = center + (pixel - 0.5 * screen) / pitch;

    if (u.x < 0.0 || u.y < 0.0 || u.x >= float(params.maze.x) || u.y >= float(params.maze.y)) {
        return kBackground;
    }

    const uint mip_count = max(1u, params.maze.w);
    const float lod = clamp(log2(max(1.0 / (pitch * 4.0), 1.0)), 0.0,
                            float(mip_count - 1u));
    const uint l0 = uint(floor(lod));
    const uint l1 = min(l0 + 1u, mip_count - 1u);
    const float blend = lod - floor(lod);

    const float d0 = SampleDensityLevel(u, l0);
    const float d1 = SampleDensityLevel(u, l1);
    const float density = lerp(d0, d1, blend);
    const float coverage = clamp(2.0 * density * params.view.w, 0.0, 1.0);
    return lerp(kFloor, kWall, coverage);
}

[shader("fragment")]
float4 main(PSInput input) : SV_TARGET {
    const uint tier = uint(params.screen.z + 0.5);
    if (tier == 0u) {
        return ExactMaze(input.position.xy);
    }
    if (tier == 1u) {
        return AnalyticMaze(input.position.xy);
    }
    return DensityMaze(input.position.xy);
}
