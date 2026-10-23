// CPU-side debug state for the solver visualization.
//
// One byte per cell stores the solver's flag bits plus the parent arrow:
//
//   bits 0..3 : Open / Closed / Path / Current (SolverCellFlag's values)
//   bit  4    : reserved
//   bits 5..6 : parent direction (0=N 1=E 2=S 3=W)
//   bit  7    : parent valid
//
// A packed per-block summary (8x8 cells, three uint8 counters) gives the
// analytic/density tiers a cheap source: the overlay shader reads one uint per
// block instead of touching the full cell grid at low zoom.
//
// The grid owns no GPU state. FlushTo() copies only the cells/blocks that
// changed since the previous flush into caller-provided mapped buffers, so a
// frame that expands a few hundred nodes does a few hundred byte writes.

module;

export module MazeGenerator.DebugGrid;

import std;

import MazeGenerator.Solver;

export namespace MazeGenerator {

class DebugGrid {
public:
    static constexpr std::uint32_t kBlockSize = 8;
    static constexpr std::uint32_t kBlockCells = kBlockSize * kBlockSize;
    static constexpr std::uint8_t kFlagMask = 0x1Fu;
    static constexpr std::uint8_t kArrowMask = 0xE0u;

    void Reset(std::uint32_t width, std::uint32_t height);
    void Clear();
    void Apply(const SolverCellChange& change);

    // `cells` is Width()*Height() bytes, `blocks` is BlocksX()*BlocksY() packed
    // counters. Null pointers are accepted (the caller may have no GPU state).
    void FlushTo(std::uint8_t* cells, std::uint32_t* blocks);

    [[nodiscard]] std::uint32_t Width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t Height() const noexcept { return height_; }
    [[nodiscard]] std::uint32_t BlocksX() const noexcept { return blocks_x_; }
    [[nodiscard]] std::uint32_t BlocksY() const noexcept { return blocks_y_; }

    [[nodiscard]] std::uint8_t Cell(std::uint32_t index) const noexcept {
        return index < cells_.size() ? cells_[index] : std::uint8_t{0};
    }
    [[nodiscard]] std::uint32_t Block(std::uint32_t index) const noexcept {
        return index < blocks_.size() ? blocks_[index] : std::uint32_t{0};
    }

private:
    void UpdateBlock(std::uint32_t index, std::uint8_t old_flags,
                     std::uint8_t new_flags);

    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t blocks_x_ = 0;
    std::uint32_t blocks_y_ = 0;
    std::vector<std::uint8_t> cells_{};
    std::vector<std::uint32_t> blocks_{};
    std::vector<std::uint32_t> dirty_cells_{};
    std::vector<std::uint32_t> dirty_blocks_{};
    bool full_dirty_ = true;
};

inline void DebugGrid::Reset(std::uint32_t width, std::uint32_t height) {
    width_ = width;
    height_ = height;
    blocks_x_ = (width + kBlockSize - 1) / kBlockSize;
    blocks_y_ = (height + kBlockSize - 1) / kBlockSize;
    cells_.assign(static_cast<std::size_t>(width_) * height_, std::uint8_t{0});
    blocks_.assign(static_cast<std::size_t>(blocks_x_) * blocks_y_,
                   std::uint32_t{0});
    dirty_cells_.clear();
    dirty_blocks_.clear();
    full_dirty_ = true;
}

inline void DebugGrid::Clear() {
    std::fill(cells_.begin(), cells_.end(), std::uint8_t{0});
    std::fill(blocks_.begin(), blocks_.end(), std::uint32_t{0});
    dirty_cells_.clear();
    dirty_blocks_.clear();
    full_dirty_ = true;
}

inline void DebugGrid::Apply(const SolverCellChange& change) {
    if (change.index >= cells_.size()) {
        return;
    }

    const std::uint8_t old_value = cells_[change.index];
    const std::uint8_t old_flags = old_value & kFlagMask;
    const std::uint8_t new_flags =
        static_cast<std::uint8_t>(change.flags) & kFlagMask;

    std::uint8_t arrow = old_value & kArrowMask;
    if (change.parent_dir != kKeepParentDir) {
        arrow = change.parent_dir == kNoParentDir
                    ? std::uint8_t{0}
                    : static_cast<std::uint8_t>(
                          0x80u | (static_cast<std::uint8_t>(change.parent_dir)
                                   << 5));
    }

    if (old_flags == new_flags && arrow == (old_value & kArrowMask)) {
        return;
    }

    cells_[change.index] = static_cast<std::uint8_t>(new_flags | arrow);
    UpdateBlock(change.index, old_flags, new_flags);
    dirty_cells_.push_back(change.index);
}

inline void DebugGrid::UpdateBlock(std::uint32_t index, std::uint8_t old_flags,
                                   std::uint8_t new_flags) {
    const std::uint32_t x = index % width_;
    const std::uint32_t y = index / width_;
    const std::uint32_t block = (y / kBlockSize) * blocks_x_ + x / kBlockSize;

    const auto has = [](std::uint8_t value, SolverCellFlag bit) -> int {
        return (value & static_cast<std::uint8_t>(bit)) != 0 ? 1 : 0;
    };
    const auto delta = [&](SolverCellFlag bit) -> int {
        return has(new_flags, bit) - has(old_flags, bit);
    };

    int open = static_cast<int>(blocks_[block] & 0xFFu);
    int closed = static_cast<int>((blocks_[block] >> 8u) & 0xFFu);
    int path = static_cast<int>((blocks_[block] >> 16u) & 0xFFu);

    open += delta(SolverCellFlag::Open);
    closed += delta(SolverCellFlag::Closed);
    path += delta(SolverCellFlag::Path);

    blocks_[block] = static_cast<std::uint32_t>(std::max(open, 0)) |
                     (static_cast<std::uint32_t>(std::max(closed, 0)) << 8u) |
                     (static_cast<std::uint32_t>(std::max(path, 0)) << 16u);
    dirty_blocks_.push_back(block);
}

inline void DebugGrid::FlushTo(std::uint8_t* cells, std::uint32_t* blocks) {
    if (full_dirty_) {
        if (cells != nullptr && !cells_.empty()) {
            std::memcpy(cells, cells_.data(), cells_.size());
        }
        if (blocks != nullptr && !blocks_.empty()) {
            std::memcpy(blocks, blocks_.data(),
                        blocks_.size() * sizeof(std::uint32_t));
        }
        full_dirty_ = false;
    } else {
        if (cells != nullptr) {
            for (const std::uint32_t index : dirty_cells_) {
                cells[index] = cells_[index];
            }
        }
        if (blocks != nullptr) {
            for (const std::uint32_t index : dirty_blocks_) {
                blocks[index] = blocks_[index];
            }
        }
    }
    dirty_cells_.clear();
    dirty_blocks_.clear();
}

} // namespace MazeGenerator
