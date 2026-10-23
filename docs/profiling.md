# Frame and GPU profiling

Two opt-in diagnostics answer different questions:

- **`--frame-profile`** measures the CPU frame loop's wall-clock time and names
  the stage (or the unaccounted wait) that made a frame long.
- **`--gpu-profile`** brackets every render-graph pass with a GPU timestamp and
  reports the per-pass mean/max over a window of frames.

Both are off by default. With the flags absent, the frame loop reads no clock
for profiling and the renderer creates no timestamp query pool and writes no
timestamps. (The existing pipeline-statistics queries are a separate, always-on
feature.)

The app binaries are multi-config; the path below assumes a RelWithDebInfo
build (the only build whose numbers are worth reading):

```bash
BIN=build/bin/examples/RelWithDebInfo/basic_scene/basic_scene

# CPU loop profile, one summary every 300 iterations.
"$BIN" --frame-profile --frame-profile-every 300 --max-frames 900

# GPU per-pass table every 120 rendered frames, with validation on.
"$BIN" --gpu-profile --gpu-profile-every 120 --max-frames 600 --validation

# They are independent and can run together.
"$BIN" --frame-profile --gpu-profile --max-frames 600
```

## CPU frame profile

### What it measures

An interval is measured from the start of one loop iteration to the start of the
next, so it includes event polling, input, acquire, the app update/render hooks,
submit, present, and any time blocked in the driver. The first iteration has no
predecessor and therefore no interval. Every stage is bracketed with a
steady-clock read only while the profiler is enabled.

| Stage | Covers |
|---|---|
| `poll-events` | `platform->PollEvents()` |
| `pre-input` | `on_pre_input` hooks |
| `input-process` | event filtering, `InputSystem::ProcessEvents`, quit checks, `NotifyWindowMinimized` |
| `runtime-begin` | `FrameLoop::BeginFrame()` only |
| `present-drain` | the deferred-present drain (DiscardStale only) |
| `bootstrap-begin` | `VulkanBootstrap::BeginFrame()` only |
| `swapchain-recreate` | `RecreateSwapchain()`, when the swapchain is out of date |
| `acquire-image` | `AcquireNextImage()` |
| `app-update` | the fixed-step loop plus `on_frame_update` |
| `app-render` | `on_frame_render` |
| `submit` | `SubmitFrame()` |
| `present` | `Present()` in the sequential policy |
| `runtime-end` | `FrameLoop::EndFrame()` |
| `bootstrap-end` | `VulkanBootstrap::EndFrame()` |
| `unaccounted` | the interval remainder no scope covered |

`acquire-image` often absorbs a vsync stall: a frame that "renders in 2 ms" but
waits 14 ms for an image is visible there, not in `app-render`. `unaccounted` is
where an unwrapped fence wait, a present, or OS descheduling lands.

### Reading the summary

```
Frame profile over 120 iterations: interval n=120 mean 9.33 ms, p50 8.63, p90 10.57, p99 45.55, min 3.57, max 59.36; 2 long frames (>= 20.0 ms)
  present interval: n=120 mean 8.89 ms, p50 8.57, p90 10.44, p99 18.71, min 3.59, max 45.57
  stages (total ms, avg ms, max ms):
    acquire-image           602.9    4.983     14.29
    app-render              347.6    2.873     43.47
    ...
    (measured)             1123.4  (sum of stages vs 1119.0 ms of interval time)
  ecs systems (a breakdown of app-update / app-render; not additional time):
    frame      frame.render_submit        n=121  total    336.0 ms  avg 2.777 ms  max 43.36 ms
    ...
  blame for the 2 long frames:
    app-render              1 frames,       43.5 ms total,    43.47 ms avg
    app-update              1 frames,       37.4 ms total,    37.38 ms avg
```

- **Interval quantiles are nearest-rank over every stored sample**, not bucket
  edges. A p99 that is a real freeze shows the freeze. Up to 100,000 samples are
  stored per window; past that the count/total/min/max stay exact and the
  summary says the distribution covers only the stored prefix.
- **`(measured)`** compares the sum of stage samples with the interval total. In
  steady state they agree closely. A window that contains a swapchain recreation
  can exceed the interval total: the recreation's stage time is deliberately
  recorded while that iteration's interval is discarded (the clock is re-stamped
  so the one-off stall is not billed to the next frame). The stage rows are
  still correct; only the remainder is not a precise unaccounted figure for that
  window.
- **`blame`** is per long frame: the single largest stage, or `unaccounted` when
  the remainder is larger. This is what turns "a 60 ms frame" into "the acquire
  was 55 ms" or "nothing we wrapped was 55 ms". Short frames are not blamed.
- **`ecs systems`** is a subset of `app-update`/`app-render`, never additional
  time. Rows are per `FrameSchedule` frame system and per `SystemSchedule` data
  system, labelled by the system's registered label, then its engine slot, then
  its registration index. Sample counts reflect executions: with fixed stepping
  a component system can run 0..N times in one rendered frame.
- **`present interval`** is the spacing between successful present calls. In the
  sequential policy that is the display cadence. Under `PresentPolicy::DiscardStale`
  several completed frames may be drained in one iteration and the Mailbox
  compositor drops superseded ones, so there it is present throughput, not the
  cadence actually scanned out.

A summary is emitted every `--frame-profile-every` iterations (default 600);
the window resets after each summary, so consecutive summaries describe disjoint
spans. A shorter run flushes one final summary at exit. `--frame-profile-long-ms`
sets the long-frame threshold (default 20 ms).

### Cost

When enabled, the profiler reads the steady clock once per stage, keeps the
interval samples, and (for the ECS table) allocates one small string per system
execution for its label. That allocation happens before the measured interval
starts. When disabled, the schedules take a single null check and read no clock.

## GPU per-pass timings

### What it measures

The renderer creates one timestamp query pool on the graphics queue family and
writes a timestamp pair around every render-graph pass that has an execute
callback. A dedicated reserved index, `frame-prep`, brackets the work recorded
ahead of the graph: staged texture uploads, descriptor writes, Hi-Z setup, and
the physical-camera composite.

```
GPU pass timing: 120 window-renders, 14 pass(es), 8.934 ms per render summed over passes
  collect                      mean   2.069 ms  max   9.993 ms  n=120
  hiz-gen                      mean   0.808 ms  max   3.564 ms  n=120
  main-pass                    mean   0.330 ms  max   0.934 ms  n=120
  frame-prep (uploads + composite) mean   0.046 ms  max   1.100 ms  n=120
  ...
```

Names come verbatim from the compiled graph, so a pass named `collect` appears as
`collect`, not `collect-pass`. Rows are sorted by total time descending; `n` is
the number of collected samples in the window. The header's ms figure is the sum
of all pass means for one render, useful as a rough frame-total check.

Samples are read only when a ring slot is reused, after the prior submission is
fence-complete (the same guarantee the pipeline-statistics pool uses). The first
frames therefore produce no table; `--gpu-profile-every` counts collections that
actually yielded samples, and the window (and its counter) resets after each log.

### What it does not measure

- **Begin/end rendering and barrier emission are outside the per-pass brackets.**
  The bracket covers the pass's execute callback, so a pass's reported cost
  excludes the `vkCmdBeginRendering`/`vkCmdEndRendering` and the planned
  barriers around it. `frame-prep` likewise covers `record_prep` only; the
  queue submission is outside.
- **Passes under about 0.05 ms are timestamp noise** on most GPUs; read the
  mean, not a single sample.
- **Async compute**: a query pool cannot span queue families. The timer is
  created on the graphics family, and graphics runs are the only ones that pass
  it through `RenderPipeline::RecordRun`. Compute and transfer passes are absent
  from the table, and an info line says so once. This is deliberate: disabling
  the whole timer would make `--gpu-profile` silent on desktop GPUs that expose
  a compute-only family.
- **No timestamp support** (`timestampValidBits == 0` on the graphics family):
  the timer logs a warning and stays disabled; rendering proceeds normally.
- **`VK_NOT_READY` / partial reads / an end that never landed**: those queries
  are skipped individually; one unready query never discards the whole table.
- **CPU-side recording cost** is not in the GPU table; use `--frame-profile`
  (or a capture) for that.

## Choosing between them

Start with `--gpu-profile` when a pass is suspected, and with `--frame-profile`
when the frame is slow but the GPU table is not. The CPU profiler is the only one
that can name a wait in acquire, present, or the driver; the GPU table is the
only one that can show a pass's actual device time. If both are clean and the
frame is still wrong, capture with RenderDoc or a vendor tool for counters and
barrier layout checks (see
[the render-graph parity checklist](render-graph-parity-checklist.md)).

The GPU flags are diagnostics, not a frame-pacing feature: they add query pairs
and a log write, and leave present policy and `delta_time` untouched.

## Related benchmark

Both profilers above measure the engine and renderer. For the packed maze
storage itself, `maze_bench` runs the lookup/set workloads headlessly on a
device-free target; see [maze-benchmark.md](maze-benchmark.md).
