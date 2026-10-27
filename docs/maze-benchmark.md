# Maze storage benchmark

`maze_bench` is a headless micro-benchmark for the packed `TiledMaze` storage
([app/TiledMaze.cppm](../app/TiledMaze.cppm)) and the density builder. It links
only the app's standalone maze modules - no engine, no Vulkan device, no window
- so it can run anywhere and in any configuration.

It is deliberately separate from the frame and GPU profilers described in
[profiling.md](profiling.md): those measure the renderer, this measures the maze
data structure.

## Build and run

```bash
cmake --build build --config RelWithDebInfo --target maze_bench
build/bin/bench/RelWithDebInfo/maze_bench
```

Read only RelWithDebInfo or Release numbers. Debug keeps the `assert`s in
`Locate`/`Bit` and compiles at `-O0`, which changes both the instruction count
and the inlining around every lookup.

## The lookup suite (default)

The default run sweeps the maze sizes `256, 1024, 4096, 16384` and reports four
workloads per size:

| Workload | Operations | What it measures |
|---|---|---|
| `sequential-sweep` | one row-major pass over every cell | best case: streaming, one pass, no reuse |
| `random-lookup` | `--ops` independent random cells | read throughput with a random footprint |
| `random-walk` | `--ops` dependent 4-neighbour steps | serialized load latency with 2D locality |
| `random-open` | `--sets` random `OpenNorth`/`OpenWest` | clear-bit write cost |

A lookup reads a cell's north and west bits, i.e. the two bits the density
builder reads. Both usually live in the same 64-byte tile line; `East`/`South`
are not part of the unit because they redirect to a neighbouring cell and can
touch a second line.

Each workload is measured once per `--repeat` (default 5) and the table reports
the median over repetitions, plus the slowest repetition. The coordinates and
walk directions for every 1024-operation block are generated *outside* the
timed region, so the numbers do not include the RNG.

### Sample output

One machine, one run (`--seed 1234`, defaults):

```
=== 4096 x 4096 | 16777216 cells | 4.031 MiB | 66049 tiles ===
workload                    ops     ns/op     Mops/s    median ms       max ms
sequential-sweep       16777216     1.200     833.10       20.138       20.487
random-lookup           1000000     6.822     146.59        6.822        7.197
random-walk             1000000     2.609     383.34        2.609        2.769
random-open              250000     3.652     273.79        0.913        1.874
```

Across the default size sweep:

| Size | Working set | sequential-sweep | random-lookup | random-walk | random-open |
|---|---|---|---|---|---|
| 256² | 0.018 MiB | 1.52 ns | 1.40 ns | 2.55 ns | 1.21 ns |
| 1024² | 0.258 MiB | 1.19 ns | 1.43 ns | 2.62 ns | 1.49 ns |
| 4096² | 4.031 MiB | 1.20 ns | 6.82 ns | 2.61 ns | 3.65 ns |
| 16384² | 64.125 MiB | 1.20 ns | 12.11 ns | 2.58 ns | 11.18 ns |

These are illustrative, not golden values: they depend on the CPU, its cache
sizes, and the memory clock.

### Reading it

- **`sequential-sweep` is the reference.** It stays at roughly 1.2 ns/cell at
  every size because it never revisits a line. A random workload that is much
  slower than this is paying for its footprint, which is the point of the
  comparison.
- **The random-lookup curve should step at the cache boundaries.** 4 MiB is the
  size the app ships (`kMazeWidth`/`kMazeHeight` in [app/App.cpp](../app/App.cpp))
  and is L3-resident on most desktop CPUs; 64 MiB forces DRAM. A flat curve
  would mean the lookups are not actually missing, or the compiler removed
  them - check the printed `checksum`.
- **`random-walk` is a latency number, not a throughput number.** Each step
  depends on the previous one, so the loads cannot overlap, while the
  independent random lookups can. Do not read a lower walk number as "the walk
  is faster": it is also spatially local, so it mostly stays in L1/L2. Compare
  it against itself across sizes, not against `random-lookup`.
- **`random-open` is a clear-bit write.** Each 1024-operation block uses one
  setter (`OpenNorth`/`OpenWest`) so an unpredictable north/west branch does not
  end up inside the measured loop. The symmetric `CloseNorth`/`CloseWest` exist
  but are not benchmarked.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--seed N` | `1234` | Seeds both streams; the two derived seeds are printed |
| `--ops N` | `1000000` | Lookups per repetition for `random-lookup` and `random-walk` |
| `--sets N` | `250000` | Wall opens per repetition |
| `--repeat N` | `5` | Repetitions per workload (1..1000) |
| `--sizes N,...` | `256,1024,4096,16384` | Maze side lengths for the sweep |
| `--generation` | off | Run only the generation suite (below) |
| `--generation-size N` | `4096` | Maze side length for the generation suite |

Fast iteration on the app's size only:

```bash
build/bin/bench/RelWithDebInfo/maze_bench --sizes 4096 --ops 4000000
```

The 16384² maze takes several seconds to generate, and the generation time is
not part of any timed workload; it dominates the wall clock of a default run.

## Reproducibility

`--seed` is split into a maze seed and a workload seed, and the workload seed
into one stream per workload. Adding a workload or changing `--ops` therefore
does not shift another workload's random sequence. Within a run every
repetition restarts its stream at the same offset, so repetitions perform
identical work. `random-open` also copies the maze before each repetition, so
the set workload never accumulates across repetitions.

The printed `checksum` is the sink that keeps the reads and writes observable.
It is not a validation of the maze contents (the unit tests cover that).

## The generation suite

`--generation` measures the other half of the maze code: the O(cells) work the
app does at startup.

```bash
build/bin/bench/RelWithDebInfo/maze_bench --generation --generation-size 4096
```

```
=== generation: 4096 x 4096 | 16777216 cells | 4.031 MiB | 11 density levels | repeat=3 ===
step                 median ms     min ms     max ms     Mcells/s
create-random          472.814    463.809    487.590         35.5
build-density           37.007     35.747     40.143        453.4
create+build           512.957    500.816    523.336         32.7
```

- `create-random` is `TiledMaze::CreateRandom`: allocation, the all-walls fill,
  and one `std::bernoulli_distribution` draw per interior wall. Its cost is part
  of what is measured, and it is by far the slower of the two steps.
- `build-density` is `BuildMazeDensity`: one lookup pair per cell plus the mip
  chain.
- `create+build` is the end-to-end startup cost.

This suite is separate because it answers a different question and, at large
sizes, is far slower than the lookup suite.

## Limitations

- Single-threaded. It does not exercise contention or parallel writers.
- No GPU and no render pass. Use `--frame-profile` / `--gpu-profile` for those.
- Only the maze API's own primitives are benchmarked; a lookup is two bit
  tests, not a full shader evaluation (which also does coordinate math and
  wall-thickness tests).
- Clock reads bracket each 1024-operation block, not each operation. The
  overhead is under 0.05 ns/op, but the report has no per-operation
  distribution - the spread shown is across repetitions.
- Timings exclude the gaps between blocks (coordinate generation), which also
  means the CPU frequency ramp is less visible than in a single long loop.
