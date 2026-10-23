# Render-graph manual parity checklist

The automated tests catch determinism and hashes. This list covers the things a
human notices. Run it by hand after any change that touches graph compilation,
barrier planning, resource aliasing, or the pass dispatch path.

Run the reference scene with validation layers and watch the log:

```bash
./build/bin/examples/Debug/basic_scene/basic_scene --validation
```

1. **Startup** — no validation errors or warnings during instance/device/swapchain
   creation; all shaders/pipelines compile first try.
2. **Steady state** — no per-frame validation messages (especially synchronization
   VUIDs) once the first frames have passed.
3. **Resize** — drag-resize the window; no `VUID-vkQueuePresentKHR`, no out-of-date
   storm, no leaked swapchain images; the image returns to a correct frame.
4. **Minimize / restore** — zero-size framebuffer is handled (no crash, no
   validation error), rendering resumes on restore.
5. **Hot reload** — edit a shader; the engine reloads it without a validation
   error and the change appears/behaves as expected.
6. **Add/remove an app pass at runtime** — insert a custom pass, confirm it runs
   in the expected slot, remove it, confirm the remaining frames are still correct.
7. **Visual correctness** — compare against the previous build: geometry, culling,
   occlusion, and post effects are pixel-equivalent (no flicker, no dark/blank
   frames after a graph mutation).
8. **Clean shutdown** — no validation errors on teardown; device drains before
   resources are freed.
9. **GPU counters (optional)** — first run the built-in per-pass timings
   (`--gpu-profile`, see [frame and GPU profiling](profiling.md)), then capture
   with RenderDoc or a vendor tool: check that barrier counts and layouts did
   not regress unexpectedly, and that alias reuse actually happens for the
   resources intended to alias.
