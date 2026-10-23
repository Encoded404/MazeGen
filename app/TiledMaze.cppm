// Standalone tiled wall storage for a 2D maze.
//
// Representation:
//   * two bits per cell: north (even bit) and west (odd bit)
//   * south is the north wall of the cell below
//   * east is the west wall of the cell to the right
//   * one extra row and column of cells hold the south/east boundary walls
//
// Physical layout:
//   * cells are grouped into 16x16 tiles
//   * each tile is eight uint64 words = 64 bytes
//   * that is one cache line on the usual 64-byte-line PC targets; on a
//     128-byte-line CPU two tiles share a line
//   * within a tile, local = (y & 15) * 16 + (x & 15)
//   * word = local / 32, bit = 2 * (local % 32)
//
// A plain vector<uint64_t> is only guaranteed 8-byte alignment, so Tile is
// alignas(64): that is what makes "one tile == one 64-byte line" real.
//
// This file is standalone for now: nothing imports TiledMaze yet.

module;

#include <cassert>

export module MazeGenerator.TiledMaze;

import std;

export namespace MazeGenerator {

class TiledMaze {
public:
    static constexpr std::uint32_t kTileShift = 4;
    static constexpr std::uint32_t kTileSize = 1U << kTileShift;             // 16
    static constexpr std::uint32_t kCellsPerTile = kTileSize * kTileSize;    // 256
    static constexpr std::uint32_t kWordsPerTile = kCellsPerTile * 2U / 64U; // 8

    // Creates a width x height maze with every wall present. The extra
    // boundary row/column is allocated internally.
    [[nodiscard]] static TiledMaze Create(std::uint32_t width, std::uint32_t height);

    // Creates a width x height maze with every border wall present except for
    // one opening in the top row and one mirrored opening in the bottom row.
    // Each interior wall is open with probability `open_probability`
    // (0 = no interior passages, 1 = no interior walls). The same seed always
    // produces the same maze.
    [[nodiscard]] static TiledMaze CreateRandom(
        std::uint32_t width,
        std::uint32_t height,
        double open_probability,
        std::uint32_t seed = std::random_device{}());

    [[nodiscard]] std::uint32_t Width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t Height() const noexcept { return height_; }

    [[nodiscard]] bool North(std::uint32_t x, std::uint32_t y) const noexcept;
    [[nodiscard]] bool West(std::uint32_t x, std::uint32_t y) const noexcept;
    [[nodiscard]] bool South(std::uint32_t x, std::uint32_t y) const noexcept;
    [[nodiscard]] bool East(std::uint32_t x, std::uint32_t y) const noexcept;

    void OpenNorth(std::uint32_t x, std::uint32_t y) noexcept;
    void OpenWest(std::uint32_t x, std::uint32_t y) noexcept;
    void OpenSouth(std::uint32_t x, std::uint32_t y) noexcept;
    void OpenEast(std::uint32_t x, std::uint32_t y) noexcept;

    [[nodiscard]] bool operator==(const TiledMaze& other) const noexcept;

private:
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    // One tile is exactly one cache line. alignas(64) is what makes that
    // guarantee hold for the vector's heap allocation.
    struct alignas(64) Tile {
        std::array<std::uint64_t, kWordsPerTile> words{};

        [[nodiscard]] friend bool operator==(const Tile& a, const Tile& b) noexcept {
            return a.words == b.words;
        }
    };

    static_assert(sizeof(Tile) == 64, "a tile must be exactly 64 bytes");
    static_assert(alignof(Tile) == 64, "a tile must start on a 64-byte boundary");
    static_assert(kWordsPerTile == 8, "16x16 cells at 2 bits each is eight words");
    static_assert(kCellsPerTile * 2 == kWordsPerTile * 64);

    // The result of mapping a cell to its tile, word, and bit offset.
    struct CellPos {
        std::size_t tile = 0;
        std::uint32_t word = 0;
        std::uint32_t shift = 0;
    };
    // NOLINTEND(misc-non-private-member-variables-in-classes)

    [[nodiscard]] CellPos Locate(std::uint32_t x, std::uint32_t y) const noexcept;
    [[nodiscard]] bool Bit(CellPos pos, std::uint32_t offset) const noexcept;
    void Clear(CellPos pos, std::uint32_t offset) noexcept;

    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::size_t tiles_x_ = 0;
    std::size_t tiles_y_ = 0;
    std::vector<Tile> tiles_;
};

inline TiledMaze TiledMaze::Create(std::uint32_t width, std::uint32_t height) {
    assert(width > 0 && height > 0);

    TiledMaze maze;
    maze.width_ = width;
    maze.height_ = height;

    // +1 for the extra boundary row/column; round up to whole tiles.
    const std::size_t cells_wide = static_cast<std::size_t>(width) + 1;
    const std::size_t cells_high = static_cast<std::size_t>(height) + 1;
    maze.tiles_x_ = (cells_wide + (kTileSize - 1)) >> kTileShift;
    maze.tiles_y_ = (cells_high + (kTileSize - 1)) >> kTileShift;

    maze.tiles_.resize(maze.tiles_x_ * maze.tiles_y_);
    for (Tile& tile : maze.tiles_) {
        tile.words.fill(~std::uint64_t{0});
    }
    return maze;
}

inline TiledMaze TiledMaze::CreateRandom(std::uint32_t width,
                                         std::uint32_t height,
                                         double open_probability,
                                         std::uint32_t seed) {
    assert(open_probability >= 0.0 && open_probability <= 1.0);

    TiledMaze maze = Create(width, height); // every wall starts closed

    std::mt19937 rng(seed);
    std::bernoulli_distribution open(open_probability);

    // Interior vertical walls: between columns x - 1 and x.
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 1; x < width; ++x) {
            if (open(rng)) {
                maze.OpenWest(x, y);
            }
        }
    }

    // Interior horizontal walls: between rows y - 1 and y.
    for (std::uint32_t y = 1; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            if (open(rng)) {
                maze.OpenNorth(x, y);
            }
        }
    }

    // One top opening, mirrored to the bottom. width - top_x would be width
    // when top_x == 0, so use the last column index for the mirror.
    std::uniform_int_distribution<std::uint32_t> top_cell(0, width - 1);
    const std::uint32_t top_x = top_cell(rng);
    const std::uint32_t bottom_x = width - 1 - top_x;

    maze.OpenNorth(top_x, 0);             // top opening
    maze.OpenSouth(bottom_x, height - 1); // bottom opening

    return maze;
}

inline TiledMaze::CellPos TiledMaze::Locate(std::uint32_t x, std::uint32_t y) const noexcept {
    // x == width_ / y == height_ are valid: they address the extra boundary
    // column/row that holds the east/south boundary walls.
    assert(x <= width_ && y <= height_);

    const std::uint32_t tile_x = x >> kTileShift; // divide by 16 to get the tile x.
    const std::uint32_t tile_y = y >> kTileShift; // divide by 16 to get the tile y.
    const std::uint32_t local_x = x & (kTileSize - 1); // 0..15 the same as modulo 15. gets the x index within the tile
    const std::uint32_t local_y = y & (kTileSize - 1); // 0..15 the same as modulo 15. gets the y index within the tile
    const std::uint32_t local = local_y * kTileSize + local_x; // 0..255 the memory index within the tile from x and y.

    const std::size_t tile =
        static_cast<std::size_t>(tile_y) * tiles_x_ + tile_x; // the index in the tile memory.
    assert(tile < tiles_.size());

    return CellPos{
        .tile = tile,
        .word = local >> 5,        // 32 cells per 64-bit word
        .shift = 2 * (local & 31), // two bits per cell, 0..62
    };
}

inline bool TiledMaze::Bit(CellPos pos, std::uint32_t offset) const noexcept {
    assert(pos.tile < tiles_.size());
    assert(pos.word < kWordsPerTile);
    assert(pos.shift + offset < 64);
    return ((tiles_[pos.tile].words[pos.word] >> (pos.shift + offset)) & 1U) != 0;
}

inline void TiledMaze::Clear(CellPos pos, std::uint32_t offset) noexcept {
    assert(pos.tile < tiles_.size());
    assert(pos.word < kWordsPerTile);
    assert(pos.shift + offset < 64);
    tiles_[pos.tile].words[pos.word] &= ~(std::uint64_t{1} << (pos.shift + offset));
}

inline bool TiledMaze::North(std::uint32_t x, std::uint32_t y) const noexcept {
    return Bit(Locate(x, y), 0U);
}

inline bool TiledMaze::West(std::uint32_t x, std::uint32_t y) const noexcept {
    return Bit(Locate(x, y), 1U);
}

inline bool TiledMaze::South(std::uint32_t x, std::uint32_t y) const noexcept {
    // The south wall is the north wall of the cell below. For y == height - 1
    // this lands in the extra boundary row.
    return North(x, y + 1);
}

inline bool TiledMaze::East(std::uint32_t x, std::uint32_t y) const noexcept {
    // The east wall is the west wall of the cell to the right. For
    // x == width - 1 this lands in the extra boundary column.
    return West(x + 1, y);
}

inline void TiledMaze::OpenNorth(std::uint32_t x, std::uint32_t y) noexcept {
    Clear(Locate(x, y), 0U);
}

inline void TiledMaze::OpenWest(std::uint32_t x, std::uint32_t y) noexcept {
    Clear(Locate(x, y), 1U);
}

inline void TiledMaze::OpenSouth(std::uint32_t x, std::uint32_t y) noexcept {
    OpenNorth(x, y + 1);
}

inline void TiledMaze::OpenEast(std::uint32_t x, std::uint32_t y) noexcept {
    OpenWest(x + 1, y);
}

inline bool TiledMaze::operator==(const TiledMaze& other) const noexcept {
    // Compare dimensions first: different dimensions can have the same total
    // bit count, so the word arrays alone are not enough.
    return width_ == other.width_ &&
           height_ == other.height_ &&
           tiles_x_ == other.tiles_x_ &&
           tiles_y_ == other.tiles_y_ &&
           tiles_ == other.tiles_;
}

} // namespace MazeGenerator
