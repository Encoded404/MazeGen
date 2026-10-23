struct DebugConstants {
    float4 view;   // center_x, center_y, pixels_per_cell, alpha
    float4 screen; // width, height, tier (0/1/2), debug block size
    uint4 grid;    // width, height, blocks_x, category mask
    uint4 extra;   // start cell index, goal cell index, flags, reserved
};

[[vk::push_constant]]
DebugConstants params;

// One byte per cell, packed four to a uint (little-endian byte order).
[[vk::binding(0, 5)]]
StructuredBuffer<uint> cellWords;

// 8x8-cell block summary, four uint8 counters packed per uint:
//   bits 0..7 open, 8..15 closed, 16..23 path, 24..31 reserved.
[[vk::binding(1, 5)]]
StructuredBuffer<uint> blockCounts;

struct PSInput {
    float4 position : SV_Position;
};

static const uint kOpen = 1u << 0;
static const uint kClosed = 1u << 1;
static const uint kPath = 1u << 2;
static const uint kCurrent = 1u << 3;
static const uint kParentValid = 1u << 7;

static const uint kCatOpen = 1u;
static const uint kCatClosed = 2u;
static const uint kCatPath = 4u;

static const uint kNoCell = 0xFFFFFFFFu;

struct CellHit {
    int2 cell;
    float2 frac;
    float2 u;
    bool inside;
};

CellHit PixelToCell(float2 pixel) {
    const float pitch = max(params.view.z, 1.0e-6);
    const float2 center = params.view.xy;
    const float2 screen = params.screen.xy;
    // Same rounded origin as ExactMaze: the overlay and the walls must use the
    // identical pixel->cell mapping or the overlay crawls while panning.
    const float2 origin = round(center * pitch - 0.5 * screen);

    CellHit hit;
    hit.u = (pixel + origin) / pitch;
    hit.cell = int2(floor(hit.u));
    hit.frac = hit.u - floor(hit.u);
    const int width = int(params.grid.x);
    const int height = int(params.grid.y);
    hit.inside = hit.cell.x >= 0 && hit.cell.y >= 0 &&
                 hit.cell.x < width && hit.cell.y < height;
    return hit;
}

uint CellFlags(uint index) {
    const uint word = cellWords[index >> 2u];
    return (word >> ((index & 3u) * 8u)) & 0xFFu;
}

float4 Palette(uint flags) {
    if ((flags & kPath) != 0u && (params.grid.w & kCatPath) != 0u) {
        return float4(0.10, 0.95, 0.55, 0.90);
    }
    if ((flags & kCurrent) != 0u) {
        return float4(1.00, 1.00, 1.00, 0.95);
    }
    if ((flags & kOpen) != 0u && (params.grid.w & kCatOpen) != 0u) {
        return float4(1.00, 0.55, 0.10, 0.70);
    }
    if ((flags & kClosed) != 0u && (params.grid.w & kCatClosed) != 0u) {
        return float4(0.20, 0.45, 1.00, 0.35);
    }
    return float4(0, 0, 0, 0);
}

uint BlockIndex(int2 block) {
    const uint blocks_x = max(params.grid.z, 1u);
    const uint blocks_y = (params.grid.y + 7u) / 8u;
    const int bx = clamp(block.x, 0, int(blocks_x) - 1);
    const int by = clamp(block.y, 0, int(blocks_y) - 1);
    return uint(by) * blocks_x + uint(bx);
}

float4 BlockColor(uint packed) {
    const float denom = 64.0; // 8x8 cells per block
    float open = float(packed & 0xFFu) / denom;
    float closed = float((packed >> 8u) & 0xFFu) / denom;
    float path = float((packed >> 16u) & 0xFFu) / denom;

    if ((params.grid.w & kCatOpen) == 0u) open = 0.0;
    if ((params.grid.w & kCatClosed) == 0u) closed = 0.0;
    if ((params.grid.w & kCatPath) == 0u) path = 0.0;

    // A one-cell path crosses a block with only ~1/64 coverage; weight it up so
    // the final path stays visible at the analytic zoom level.
    const float path_weight = path * 4.0;
    const float weight = open + closed + path_weight;
    if (weight <= 0.0) {
        return float4(0, 0, 0, 0);
    }

    const float3 rgb = (float3(1.00, 0.55, 0.10) * open +
                        float3(0.20, 0.45, 1.00) * closed +
                        float3(0.10, 0.95, 0.55) * path_weight) / weight;
    float alpha = saturate(weight * 1.6) * 0.80;
    if (uint(params.screen.z + 0.5) >= 2u) {
        alpha *= 0.55; // density tier: keep the maze readable
    }
    return float4(rgb, alpha);
}

float4 BlockOverlay(float2 u) {
    const float block_size = max(params.screen.w, 1.0);
    const float2 block_pos = u / block_size;
    const int2 b0 = int2(floor(block_pos));
    const float2 t = block_pos - floor(block_pos);

    const float4 c00 = BlockColor(blockCounts[BlockIndex(b0)]);
    const float4 c10 = BlockColor(blockCounts[BlockIndex(b0 + int2(1, 0))]);
    const float4 c01 = BlockColor(blockCounts[BlockIndex(b0 + int2(0, 1))]);
    const float4 c11 = BlockColor(blockCounts[BlockIndex(b0 + int2(1, 1))]);

    const float4 top = lerp(c00, c10, t.x);
    const float4 bottom = lerp(c01, c11, t.x);
    return lerp(top, bottom, t.y);
}

// Parent arrow: a small triangle pointing from the cell centre toward the
// parent. Drawn only when a cell is large enough to make it readable.
float4 ParentArrow(float2 pixel, int2 cell, uint flags) {
    if ((flags & kParentValid) == 0u) {
        return float4(0, 0, 0, 0);
    }

    const uint dir = (flags >> 5u) & 3u;
    float2 direction;
    if (dir == 0u) {
        direction = float2(0.0, -1.0);
    } else if (dir == 1u) {
        direction = float2(1.0, 0.0);
    } else if (dir == 2u) {
        direction = float2(0.0, 1.0);
    } else {
        direction = float2(-1.0, 0.0);
    }

    const float pitch = max(params.view.z, 1.0e-6);
    const float2 origin = round(params.view.xy * pitch - 0.5 * params.screen.xy);
    const float2 center = (float2(cell) + 0.5) * pitch - origin;
    const float2 q = (pixel - center) / pitch;

    const float forward = dot(q, direction);
    const float side = dot(q, float2(-direction.y, direction.x));
    const float tip = 0.34;
    const float base = -0.08;
    if (forward < base || forward > tip) {
        return float4(0, 0, 0, 0);
    }
    const float t = (tip - forward) / (tip - base);
    if (abs(side) > 0.13 * t) {
        return float4(0, 0, 0, 0);
    }
    return float4(1.00, 0.86, 0.35, 0.90);
}

// Constant-screen-size diamond ring at a cell centre. Used for start/goal so
// they stay visible at every LoD.
float4 CellMarker(float2 pixel, uint cell_index, float4 color) {
    if (cell_index == kNoCell || params.grid.x == 0u) {
        return float4(0, 0, 0, 0);
    }

    const int2 cell = int2(int(cell_index % params.grid.x),
                           int(cell_index / params.grid.x));
    const float pitch = max(params.view.z, 1.0e-6);
    const float2 origin = round(params.view.xy * pitch - 0.5 * params.screen.xy);
    const float2 center = (float2(cell) + 0.5) * pitch - origin;
    const float2 q = pixel - center;

    const float diamond = abs(q.x) + abs(q.y);
    if (abs(diamond - 7.0) < 1.5) {
        return color;
    }
    return float4(0, 0, 0, 0);
}

[shader("fragment")]
float4 main(PSInput input) : SV_TARGET {
    const uint tier = uint(params.screen.z + 0.5);
    const CellHit hit = PixelToCell(input.position.xy);
    float4 color = float4(0, 0, 0, 0);

    if (hit.inside) {
        if (tier == 0u) {
            const uint index =
                uint(hit.cell.y) * params.grid.x + uint(hit.cell.x);
            const uint flags = CellFlags(index);
            color = Palette(flags);

            if (color.a > 0.0) {
                if (params.view.z >= 4.0 && (flags & kClosed) != 0u) {
                    const float edge =
                        min(min(hit.frac.x, 1.0 - hit.frac.x),
                            min(hit.frac.y, 1.0 - hit.frac.y)) *
                        params.view.z;
                    if (edge < 1.0) {
                        color = lerp(color, float4(0.70, 0.85, 1.0, 0.95),
                                     0.45);
                    }
                }
                if (params.view.z >= 8.0) {
                    const float4 arrow =
                        ParentArrow(input.position.xy, hit.cell, flags);
                    if (arrow.a > 0.0) {
                        color = lerp(color, arrow, arrow.a);
                    }
                }
            }
        } else {
            color = BlockOverlay(hit.u);
        }
    }

    color.a *= params.view.w;

    const float4 start_marker = CellMarker(
        input.position.xy, params.extra.x, float4(0.10, 1.00, 0.90, 0.95));
    if (start_marker.a > 0.0) {
        color = lerp(color, start_marker, start_marker.a);
    }

    const float4 goal_marker = CellMarker(
        input.position.xy, params.extra.y, float4(1.00, 0.25, 0.55, 0.95));
    if (goal_marker.a > 0.0) {
        color = lerp(color, goal_marker, goal_marker.a);
    }

    return color;
}
