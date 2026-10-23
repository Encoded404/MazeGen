// Headless micro-benchmark for the packed TiledMaze storage.
//
// Two suites:
//   * the default lookup suite: a sequential sweep, independent random cell
//     lookups, a dependent 4-neighbour walk and random wall opens, swept over
//     a list of maze sizes so the cache hierarchy is visible;
//   * --generation: CreateRandom and BuildMazeDensity, the O(cells) work the
//     app pays at startup.
//
// Everything runs on one thread. Coordinates and directions for each block are
// generated outside the timed region, so the numbers measure the maze and its
// memory layout rather than the RNG. Each repetition starts from the same seed
// and (for the set workload) from the same maze, so repetitions are comparable.

#include <CLI/CLI.hpp>
#include <cassert>

import std;

import MazeGenerator.MazeDensity;
import MazeGenerator.TiledMaze;

namespace {

using Clock = std::chrono::steady_clock;
using MazeGenerator::BuildMazeDensity;
using MazeGenerator::TiledMaze;

constexpr double kOpenProbability = 0.45;

// One timed block. 1024 packed cells is 8 KiB, which still fits in L1 next to
// the maze lines it drives; a bigger block would evict part of the working set
// between blocks and bias the small-maze numbers.
constexpr std::size_t kBlockOps = 1024;

[[nodiscard]] double SecondsBetween(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

// Small, fast, seedable PRNG. std::mt19937 is used by CreateRandom; the
// workload streams are deliberately separate so a workload's coordinates do
// not depend on how many draws maze generation happened to consume.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

    [[nodiscard]] std::uint64_t Next() noexcept {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }

private:
    std::uint64_t state_;
};

struct Options {
    std::uint32_t seed = 1234;
    std::uint64_t ops = 1'000'000;
    std::uint64_t set_ops = 250'000;
    std::uint32_t repeat = 5;
    std::vector<std::uint32_t> sizes{256, 1024, 4096, 16384};
    std::uint32_t generation_size = 4096;
    bool generation = false;
};

struct Summary {
    double min_seconds = 0.0;
    double median_seconds = 0.0;
    double max_seconds = 0.0;
};

[[nodiscard]] Summary Summarize(std::vector<double> samples) {
    assert(!samples.empty());
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 == 0
                              ? 0.5 * (samples[middle - 1] + samples[middle])
                              : samples[middle];
    return Summary{
        .min_seconds = samples.front(),
        .median_seconds = median,
        .max_seconds = samples.back(),
    };
}

// Times `ops` operations as a sequence of blocks. `prepare` fills the block's
// pre-generated coordinates outside the clock; `run` consumes it inside. The
// per-block clock reads cost ~40 ns per 1024 ops, i.e. well under 0.05 ns/op.
[[nodiscard]] double TimeBlocks(std::uint64_t ops,
                                const std::function<void(std::size_t)>& prepare,
                                const std::function<void(std::size_t)>& run) {
    double total = 0.0;
    for (std::uint64_t done = 0; done < ops;) {
        const auto count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kBlockOps, ops - done));
        prepare(count);
        const auto start = Clock::now();
        run(count);
        total += SecondsBetween(start, Clock::now());
        done += count;
    }
    return total;
}

// Row helpers. The first table reports per-operation cost, the second reports
// whole-run wall time.
void PrintOpHeader() {
    std::cout << std::format("{:<18} {:>12} {:>9} {:>10} {:>12} {:>12}\n",
                             "workload", "ops", "ns/op", "Mops/s", "median ms", "max ms");
}

void PrintOpRow(std::string_view name, std::uint64_t ops, const Summary& summary) {
    const double ns_per_op =
        ops == 0 ? 0.0 : summary.median_seconds * 1.0e9 / static_cast<double>(ops);
    const double mops =
        summary.median_seconds == 0.0 ? 0.0 : static_cast<double>(ops) / summary.median_seconds / 1.0e6;
    std::cout << std::format("{:<18} {:>12} {:>9.3f} {:>10.2f} {:>12.3f} {:>12.3f}\n",
                             name, ops, ns_per_op, mops, summary.median_seconds * 1.0e3,
                             summary.max_seconds * 1.0e3);
}

void PrintTimeHeader() {
    std::cout << std::format("{:<18} {:>11} {:>10} {:>10} {:>12}\n",
                             "step", "median ms", "min ms", "max ms", "Mcells/s");
}

void PrintTimeRow(std::string_view name, std::uint64_t cells, const Summary& summary) {
    const double mcells =
        summary.median_seconds == 0.0
            ? 0.0
            : static_cast<double>(cells) / summary.median_seconds / 1.0e6;
    std::cout << std::format("{:<18} {:>11.3f} {:>10.3f} {:>10.3f} {:>12.1f}\n",
                             name, summary.median_seconds * 1.0e3, summary.min_seconds * 1.0e3,
                             summary.max_seconds * 1.0e3, mcells);
}

// Best case: every cell once, row-major. This is the reference the random
// numbers are read against, and it is the same walk BuildMazeDensity performs.
[[nodiscard]] std::vector<double> MeasureSweep(const TiledMaze& maze,
                                               std::uint32_t repeat,
                                               std::uint64_t& sink) {
    const std::uint32_t width = maze.Width();
    const std::uint32_t height = maze.Height();
    std::vector<double> samples(repeat);

    for (std::uint32_t r = 0; r < repeat; ++r) {
        std::uint64_t checksum = 0;
        const auto start = Clock::now();
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                checksum += static_cast<std::uint64_t>(maze.North(x, y)) +
                            2U * static_cast<std::uint64_t>(maze.West(x, y));
            }
        }
        samples[r] = SecondsBetween(start, Clock::now());
        sink ^= checksum;
    }
    return samples;
}

// Independent random cell lookups with pre-generated coordinates: the CPU can
// overlap the loads, so this is throughput with a random footprint.
[[nodiscard]] std::vector<double> MeasureRandomLookup(const TiledMaze& maze,
                                                      std::uint64_t seed,
                                                      std::uint64_t ops,
                                                      std::uint32_t repeat,
                                                      std::uint64_t& sink) {
    const std::uint64_t width = maze.Width();
    const std::uint64_t cell_count = width * maze.Height();
    std::vector<std::uint64_t> cells(kBlockOps);
    std::vector<double> samples(repeat);

    for (std::uint32_t r = 0; r < repeat; ++r) {
        SplitMix64 rng{seed + r}; // identical work in every repetition
        samples[r] = TimeBlocks(
            ops,
            [&](std::size_t count) {
                for (std::size_t i = 0; i < count; ++i) {
                    const std::uint64_t index = rng.Next() % cell_count;
                    const std::uint64_t y = index / width;
                    const std::uint64_t x = index - y * width;
                    cells[i] = (y << 32U) | x;
                }
            },
            [&](std::size_t count) {
                for (std::size_t i = 0; i < count; ++i) {
                    const auto x = static_cast<std::uint32_t>(cells[i]);
                    const auto y = static_cast<std::uint32_t>(cells[i] >> 32U);
                    sink += static_cast<std::uint64_t>(maze.North(x, y)) +
                            2U * static_cast<std::uint64_t>(maze.West(x, y));
                }
            });
    }
    return samples;
}

// Dependent 4-neighbour walk: each step's coordinates come from the previous
// step, so the loads cannot overlap. This is latency plus 2D locality, not the
// same thing as the independent random test.
[[nodiscard]] std::vector<double> MeasureWalk(const TiledMaze& maze,
                                              std::uint64_t seed,
                                              std::uint64_t ops,
                                              std::uint32_t repeat,
                                              std::uint64_t& sink) {
    const auto width = static_cast<std::int32_t>(maze.Width());
    const auto height = static_cast<std::int32_t>(maze.Height());
    std::vector<std::uint8_t> directions(kBlockOps);
    std::vector<double> samples(repeat);

    for (std::uint32_t r = 0; r < repeat; ++r) {
        SplitMix64 rng{seed + r};
        auto x = static_cast<std::int32_t>(rng.Next() % maze.Width());
        auto y = static_cast<std::int32_t>(rng.Next() % maze.Height());

        samples[r] = TimeBlocks(
            ops,
            [&](std::size_t count) {
                for (std::size_t i = 0; i < count; ++i) {
                    directions[i] = static_cast<std::uint8_t>(rng.Next() & 3U);
                }
            },
            [&](std::size_t count) {
                for (std::size_t i = 0; i < count; ++i) {
                    const auto d = static_cast<std::uint32_t>(directions[i]);
                    // Branchless step: the direction is random, so a switch
                    // would add an unpredictable branch to every step.
                    const std::int32_t dx = static_cast<std::int32_t>(d == 3U) -
                                            static_cast<std::int32_t>(d == 2U);
                    const std::int32_t dy = static_cast<std::int32_t>(d == 1U) -
                                            static_cast<std::int32_t>(d == 0U);
                    x = std::clamp(x + dx, 0, width - 1);
                    y = std::clamp(y + dy, 0, height - 1);
                    sink += static_cast<std::uint64_t>(maze.North(static_cast<std::uint32_t>(x),
                                                                  static_cast<std::uint32_t>(y))) +
                            2U * static_cast<std::uint64_t>(
                                     maze.West(static_cast<std::uint32_t>(x),
                                               static_cast<std::uint32_t>(y)));
                }
            });
    }
    return samples;
}

// Random wall opens (the API only has "open", so this is a clear-bit write).
// Each block uses a single setter: a per-op north/west branch would be 50/50
// unpredictable and dominate the measurement.
[[nodiscard]] std::vector<double> MeasureRandomOpen(const TiledMaze& base,
                                                    std::uint64_t seed,
                                                    std::uint64_t ops,
                                                    std::uint32_t repeat,
                                                    std::uint64_t& sink) {
    const std::uint64_t width = base.Width();
    const std::uint64_t cell_count = width * base.Height();
    std::vector<std::uint64_t> cells(kBlockOps);
    std::vector<double> samples(repeat);

    for (std::uint32_t r = 0; r < repeat; ++r) {
        TiledMaze maze = base; // same starting state every repetition
        SplitMix64 rng{seed + r};
        bool open_north = true;

        samples[r] = TimeBlocks(
            ops,
            [&](std::size_t count) {
                open_north = (rng.Next() & 1U) != 0U;
                for (std::size_t i = 0; i < count; ++i) {
                    const std::uint64_t index = rng.Next() % cell_count;
                    const std::uint64_t y = index / width;
                    const std::uint64_t x = index - y * width;
                    cells[i] = (y << 32U) | x;
                }
            },
            [&](std::size_t count) {
                if (open_north) {
                    for (std::size_t i = 0; i < count; ++i) {
                        maze.OpenNorth(static_cast<std::uint32_t>(cells[i]),
                                       static_cast<std::uint32_t>(cells[i] >> 32U));
                    }
                } else {
                    for (std::size_t i = 0; i < count; ++i) {
                        maze.OpenWest(static_cast<std::uint32_t>(cells[i]),
                                      static_cast<std::uint32_t>(cells[i] >> 32U));
                    }
                }
            });

        // Untimed read-back of the last block. The store indices are dynamic,
        // so the compiler cannot prove these reads do not alias any of them and
        // must keep every store in the timed loop.
        for (const std::uint64_t packed : cells) {
            const auto x = static_cast<std::uint32_t>(packed);
            const auto y = static_cast<std::uint32_t>(packed >> 32U);
            sink += static_cast<std::uint64_t>(maze.North(x, y)) +
                    2U * static_cast<std::uint64_t>(maze.West(x, y));
        }
    }
    return samples;
}

void RunGenerationSuite(const Options& options, std::uint32_t maze_seed) {
    const std::uint32_t size = options.generation_size;
    const std::uint64_t cells = static_cast<std::uint64_t>(size) * size;

    std::vector<double> create(options.repeat);
    std::vector<double> density(options.repeat);
    std::vector<double> combined(options.repeat);
    std::uint32_t levels = 0;
    std::size_t bytes = 0;
    std::uint64_t sink = 0;

    for (std::uint32_t r = 0; r < options.repeat; ++r) {
        const auto start = Clock::now();
        TiledMaze maze = TiledMaze::CreateRandom(size, size, kOpenProbability, maze_seed);
        const auto created = Clock::now();
        const auto density_data = BuildMazeDensity(maze);
        const auto built = Clock::now();

        create[r] = SecondsBetween(start, created);
        density[r] = SecondsBetween(created, built);
        combined[r] = SecondsBetween(start, built);
        levels = density_data.mip_count;
        bytes = maze.RawBytes().size();
        if (!density_data.values.empty()) {
            sink += static_cast<std::uint64_t>(density_data.values.front() * 1000.0f);
        }
    }

    std::cout << std::format("\n=== generation: {} x {} | {} cells | {:.3f} MiB | "
                             "{} density levels | repeat={} ===\n",
                             size, size, cells,
                             static_cast<double>(bytes) / (1024.0 * 1024.0), levels,
                             options.repeat);
    PrintTimeHeader();
    PrintTimeRow("create-random", cells, Summarize(create));
    PrintTimeRow("build-density", cells, Summarize(density));
    PrintTimeRow("create+build", cells, Summarize(combined));
    std::cout << std::format("checksum=0x{:016x}\n", sink);
}

} // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"Headless TiledMaze lookup/set benchmark"};
    Options options;

    app.add_option("--seed", options.seed,
                   "Seed for maze generation and workloads; both stream seeds are derived from it")
        ->type_name("N");
    app.add_option("--ops", options.ops,
                   "Lookups per repetition for the random and walk workloads")
        ->type_name("N")
        ->check(CLI::Range(std::uint64_t{1}, std::uint64_t{1} << 40U));
    app.add_option("--sets", options.set_ops, "Random wall opens per repetition")
        ->type_name("N")
        ->check(CLI::Range(std::uint64_t{1}, std::uint64_t{1} << 40U));
    app.add_option("--repeat", options.repeat, "Repetitions per workload")
        ->type_name("N")
        ->check(CLI::Range(1U, 1000U));
    app.add_option("--sizes", options.sizes,
                   "Maze side lengths to sweep (each is generated once per size)")
        ->type_name("N,...")
        ->delimiter(',')
        ->check(CLI::Range(std::uint32_t{1}, std::uint32_t{1} << 16U));
    app.add_flag("--generation", options.generation,
                 "Run only the CreateRandom/BuildMazeDensity suite");
    app.add_option("--generation-size", options.generation_size,
                   "Maze side length for the generation suite")
        ->type_name("N")
        ->check(CLI::Range(std::uint32_t{1}, std::uint32_t{1} << 16U));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    // Separate streams: the lookup coordinates do not depend on any other
    // workload, and a changing --ops cannot shift the random sequence.
    SplitMix64 seeder{options.seed};
    const auto maze_seed = static_cast<std::uint32_t>(seeder.Next());
    const std::uint64_t workload_seed = seeder.Next();
    SplitMix64 workload{workload_seed};
    const std::uint64_t lookup_seed = workload.Next();
    const std::uint64_t walk_seed = workload.Next();
    const std::uint64_t open_seed = workload.Next();

    std::cout << std::format(
        "maze_bench: seed={} maze_seed={} workload_seed=0x{:016x}\n"
        "  ops={} set_ops={} repeat={} sizes=",
        options.seed, maze_seed, workload_seed, options.ops, options.set_ops, options.repeat);
    for (std::size_t i = 0; i < options.sizes.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << options.sizes[i];
    }
    std::cout << '\n';

    if (options.generation) {
        RunGenerationSuite(options, maze_seed);
        return 0;
    }

    std::uint64_t sink = 0;

    for (const std::uint32_t size : options.sizes) {
        std::cerr << std::format("maze_bench: building {}x{} maze\n", size, size);
        const TiledMaze maze = TiledMaze::CreateRandom(size, size, kOpenProbability, maze_seed);

        const double bytes = static_cast<double>(maze.RawBytes().size());
        std::cout << std::format("\n=== {} x {} | {} cells | {:.3f} MiB | {} tiles ===\n",
                                 size, size,
                                 static_cast<std::uint64_t>(size) * size,
                                 bytes / (1024.0 * 1024.0),
                                 static_cast<std::uint64_t>(maze.TilesX()) * maze.TilesY());
        PrintOpHeader();

        PrintOpRow("sequential-sweep",
                   static_cast<std::uint64_t>(size) * size,
                   Summarize(MeasureSweep(maze, options.repeat, sink)));
        PrintOpRow("random-lookup", options.ops,
                   Summarize(MeasureRandomLookup(maze, lookup_seed, options.ops,
                                                 options.repeat, sink)));
        PrintOpRow("random-walk", options.ops,
                   Summarize(MeasureWalk(maze, walk_seed, options.ops, options.repeat, sink)));
        PrintOpRow("random-open", options.set_ops,
                   Summarize(MeasureRandomOpen(maze, open_seed, options.set_ops,
                                               options.repeat, sink)));
    }

    std::cout << std::format(
        "\nnotes:\n"
        "  sequential-sweep  best case: one row-major pass over every cell.\n"
        "  random-lookup     independent pre-generated cells (overlapped loads).\n"
        "  random-walk       dependent 4-neighbour chain (serialized loads, local).\n"
        "  random-open       clear-bit writes; one setter per 1024-op block.\n"
        "checksum=0x{:016x}\n",
        sink);
    return 0;
}
