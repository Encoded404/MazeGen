// Fixed-width bit fields per cell, packed into 64-bit words.
//
// This is algorithm scratch, not maze content: it has no extra boundary
// row/column and is never uploaded to the GPU. The width is a compile-time
// power of two so one cell never straddles a word:
//
//   * 1 bit  -> visited / unvisited
//   * 2 bits -> unseen / open / closed (A*, BFS, Dijkstra)
//   * 4 bits -> up to 16 states, e.g. frontier + visited + direction
//
// An 8-bit cell is just a byte array; keep using std::vector<std::uint8_t> for
// that rather than paying for the shift/mask in this layout.
//
// Physical layout: row-major, local index = y * width + x, word = index / cells
// per word, shift = (index % cells per word) * bits. A plain vector<uint64_t>
// is 8-byte aligned, which is all the bit operations need; the 64-byte tile
// alignment in TiledMaze is for the shader's wall layout, not for this grid.
//
// This is single-writer scratch: Set() is a read-modify-write of a 64-bit word,
// so concurrent writers to different cells in the same word would lose
// updates. Partition words per thread or use atomics if that changes.

module;

#include <cassert>
#include <logging/logging_macros.hpp>

export module MazeGenerator.PackedCellGrid;

import std;

import logiface;

export namespace MazeGenerator {

template <std::uint32_t BitsPerCell>
class PackedCellGrid {
public:
    static_assert(BitsPerCell == 1 || BitsPerCell == 2 || BitsPerCell == 4,
                  "round the cell width up to 1, 2, or 4 bits per cell");

    static constexpr std::uint32_t kBitsPerCell = BitsPerCell;
    static constexpr std::uint32_t kCellsPerWord = 64 / BitsPerCell;
    static constexpr std::uint64_t kCellMask =
        (std::uint64_t{1} << BitsPerCell) - 1;

    // Creates a width x height grid with every cell zeroed.
    [[nodiscard]] static PackedCellGrid Create(std::uint32_t width,
                                               std::uint32_t height) {
        PackedCellGrid grid;
        grid.Reset(width, height);
        return grid;
    }

    // Resizes in place (reusing the allocation when it is large enough) and
    // zeroes every cell.
    void Reset(std::uint32_t width, std::uint32_t height) {
        assert(width > 0 && height > 0);
        width_ = width;
        height_ = height;
        const std::size_t cells =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        words_.assign((cells + kCellsPerWord - 1) / kCellsPerWord, 0);
    }

    void Clear() noexcept { std::fill(words_.begin(), words_.end(), 0); }

    [[nodiscard]] std::uint32_t Width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t Height() const noexcept { return height_; }
    [[nodiscard]] std::size_t CellCount() const noexcept {
        return static_cast<std::size_t>(width_) * height_;
    }
    [[nodiscard]] std::size_t WordCount() const noexcept {
        return words_.size();
    }

    // Linear access, for solvers that already store a CellIndex.
    [[nodiscard]] std::uint32_t Get(std::size_t index) const noexcept {
        assert(index < CellCount());
        const std::size_t word = index / kCellsPerWord;
        const std::uint32_t shift =
            static_cast<std::uint32_t>(index % kCellsPerWord) * BitsPerCell;
        return static_cast<std::uint32_t>((words_[word] >> shift) & kCellMask);
    }

    void Set(std::size_t index, std::uint32_t value) noexcept {
        assert(index < CellCount());
        assert(value <= kCellMask);
        const std::size_t word = index / kCellsPerWord;
        const std::uint32_t shift =
            static_cast<std::uint32_t>(index % kCellsPerWord) * BitsPerCell;
        const std::uint64_t field = kCellMask << shift;
        words_[word] = (words_[word] & ~field) |
                       ((static_cast<std::uint64_t>(value) & kCellMask) << shift);
    }

    // Raw view of one whole 64-bit word, exactly as stored, with no per-cell
    // shift or mask. This is the fast path for "not visited" sweeps: on a
    // 1-bit grid a zero word means all 64 of its cells are still unvisited, so
    // a whole word can be rejected with a single comparison instead of 64
    // Test() calls. On wider grids a zero word likewise means every cell in it
    // is in the zero state. Cells past CellCount() in the last word are never
    // written, so those padding bits stay zero and cannot make a fully
    // unvisited word look visited. Iterate 0 .. WordCount() - 1; words are in
    // the same row-major order as the cell indices.
    [[nodiscard]] std::uint64_t Word(std::size_t word_index) const noexcept {
        assert(word_index < words_.size());
        return words_[word_index];
    }

    // True when any bit of the cell is set. For a 1-bit grid this is the
    // visited test; for wider grids it means "not in the zero state".
    [[nodiscard]] bool Test(std::size_t index) const noexcept {
        return Get(index) != 0;
    }

    // 2-D access, for generators that walk in x/y.
    [[nodiscard]] std::uint32_t Get(std::uint32_t x,
                                    std::uint32_t y) const noexcept {
        assert((x < width_ && y < height_) ||
               ReportOutOfBounds(x, y, width_, height_));
        return Get(static_cast<std::size_t>(y) * width_ + x);
    }

    void Set(std::uint32_t x, std::uint32_t y, std::uint32_t value) noexcept {
        assert((x < width_ && y < height_) ||
               ReportOutOfBounds(x, y, width_, height_));
        Set(static_cast<std::size_t>(y) * width_ + x, value);
    }

    [[nodiscard]] bool Test(std::uint32_t x, std::uint32_t y) const noexcept {
        assert((x < width_ && y < height_) ||
               ReportOutOfBounds(x, y, width_, height_));
        return Test(static_cast<std::size_t>(y) * width_ + x);
    }

    [[nodiscard]] bool operator==(const PackedCellGrid&) const = default;

private:
    // assert() only stringifies its expression, so it cannot show the runtime
    // values. Log the offending coordinate and the grid extent first, then
    // return false so assert still reports the usual file/line and aborts. The
    // whole expression (including this call) compiles out under NDEBUG.
    static bool ReportOutOfBounds(std::uint32_t x, std::uint32_t y,
                                  std::uint32_t width,
                                  std::uint32_t height) noexcept {
        LOGIFACE_LOG(error,
                     std::format("PackedCellGrid<{}>: cell (x={}, y={}) is "
                                 "outside the {}x{} grid",
                                 BitsPerCell, x, y, width, height));
        return false;
    }

    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::vector<std::uint64_t> words_{};
};

// Domain aliases: each algorithm names the state width it needs, while the
// storage stays generic.
using VisitedGrid = PackedCellGrid<1>;
using SolverStateGrid = PackedCellGrid<2>;

} // namespace MazeGenerator
