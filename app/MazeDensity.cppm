// Coarse wall-density pyramid for the far-zoom maze tier.
//
// The base level stores the average number of wall slots per cell in each
// 4x4-cell block. The shader turns that fraction back into screen coverage with
// the physical wall thickness. Levels are concatenated in one flat buffer.

module;

#include <cassert>

export module MazeGenerator.MazeDensity;

import std;

import MazeGenerator.TiledMaze;

export namespace MazeGenerator {

inline constexpr std::uint32_t kDensityBlockSize = 4;

struct MazeDensityData {
    std::vector<float> values{};
    std::uint32_t base_width = 0;
    std::uint32_t base_height = 0;
    std::uint32_t mip_count = 0;
};

// Builds the base density level plus a full 2x2-averaged mip chain. Edge blocks
// include the south/east boundary walls that live in TiledMaze's extra row and
// column, so all four maze borders contribute at far zoom.
[[nodiscard]] inline MazeDensityData BuildMazeDensity(
    const TiledMaze& maze,
    std::uint32_t block_size = kDensityBlockSize) {
    assert(block_size > 0);
    assert(maze.Width() > 0 && maze.Height() > 0);

    const std::uint32_t width = maze.Width();
    const std::uint32_t height = maze.Height();
    const std::uint32_t base_width = (width + block_size - 1) / block_size;
    const std::uint32_t base_height = (height + block_size - 1) / block_size;

    std::vector<std::uint32_t> wall_slots(base_width * base_height, 0);
    std::vector<std::uint32_t> cell_counts(base_width * base_height, 0);

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint32_t block = (y / block_size) * base_width + x / block_size;
            const std::uint32_t slots =
                (maze.North(x, y) ? 1U : 0U) + (maze.West(x, y) ? 1U : 0U);
            wall_slots[block] += slots;
            ++cell_counts[block];
        }
    }

    // The bottom and right outer walls live in the extra boundary row/column.
    // Assign each to the edge block that owns the adjacent interior cell.
    for (std::uint32_t x = 0; x < width; ++x) {
        const std::uint32_t block = ((height - 1) / block_size) * base_width + x / block_size;
        wall_slots[block] += maze.North(x, height) ? 1U : 0U;
    }
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t block = (y / block_size) * base_width + (width - 1) / block_size;
        wall_slots[block] += maze.West(width, y) ? 1U : 0U;
    }

    MazeDensityData result{};
    result.base_width = base_width;
    result.base_height = base_height;
    result.mip_count = 1;
    result.values.resize(static_cast<std::size_t>(base_width) * base_height);
    for (std::size_t i = 0; i < result.values.size(); ++i) {
        const float denominator = 2.0f * static_cast<float>(cell_counts[i]);
        result.values[i] = denominator > 0.0f
                               ? static_cast<float>(wall_slots[i]) / denominator
                               : 0.0f;
    }

    std::uint32_t level_width = base_width;
    std::uint32_t level_height = base_height;
    std::size_t level_offset = 0;
    while (level_width > 1 || level_height > 1) {
        const std::uint32_t next_width = (level_width + 1) / 2;
        const std::uint32_t next_height = (level_height + 1) / 2;
        const std::size_t next_offset = result.values.size();
        result.values.resize(next_offset + static_cast<std::size_t>(next_width) * next_height);

        for (std::uint32_t y = 0; y < next_height; ++y) {
            for (std::uint32_t x = 0; x < next_width; ++x) {
                float sum = 0.0f;
                std::uint32_t samples = 0;
                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        const std::uint32_t sx = 2 * x + dx;
                        const std::uint32_t sy = 2 * y + dy;
                        if (sx < level_width && sy < level_height) {
                            sum += result.values[level_offset +
                                                 static_cast<std::size_t>(sy) * level_width + sx];
                            ++samples;
                        }
                    }
                }
                result.values[next_offset + static_cast<std::size_t>(y) * next_width + x] =
                    static_cast<float>(sum) / static_cast<float>(samples);
            }
        }

        level_offset = next_offset;
        level_width = next_width;
        level_height = next_height;
        ++result.mip_count;
    }

    return result;
}

} // namespace MazeGenerator
