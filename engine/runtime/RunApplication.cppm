module;

#include <SDL3/SDL_video.h>

#include <logging/logging_macros.hpp>

#include "engine/core/bootstrap/Crash.hpp"

export module Runtime.Application;

import std;

import logiface;

import VulkanBackend.Event;
import VulkanBackend.Platform.SdlPlatform;
import VulkanBackend.Platform.SdlPlatformBackend;
import VulkanBackend.Vulkan.FrameLoop;
import VulkanShared.CallbackList;
import VulkanShared.ScopedSection;
import VulkanShared.Storage;
import VulkanShared.SystemProfile;
import VulkanShared.Timer;
import VulkanShared.UserPaths;
import VulkanEngine.Input;
import VulkanBackend.Vulkan.VulkanBootstrap;
import VulkanBackend.Vulkan.VulkanBootstrapBackend;
import VulkanEngine.Startup;
import VulkanEngine.Application;
import Runtime.FrameStepClock;
import Runtime.FrameProfiler;

#ifndef UINT32_MAX
constexpr std::uint32_t UINT32_MAX =
    std::numeric_limits<std::uint32_t>::max();
#endif

export namespace VulkanEngine::Application {

// NOLINTBEGIN
[[nodiscard]] inline int RunApplication(const ApplicationConfig& config, const ApplicationHooks& hooks) {
    std::unique_ptr<VulkanBackend::Platform::SdlPlatform> platform{};
    std::shared_ptr<VulkanBackend::Vulkan::IVulkanBootstrap> vk_backend{};
    std::unique_ptr<VulkanBackend::Vulkan::VulkanBootstrap> bootstrap{};
    std::unique_ptr<VulkanBackend::Vulkan::FrameLoop> runtime{};
    // Resolved once, before anything downstream needs a writable directory, and
    // held for the whole run so ApplicationContext can point at it. Declared out
    // here rather than inside the try so its lifetime does not depend on where
    // the setup steps end up.
    std::optional<VulkanShared::Storage::Storage> storage{};
    VulkanEngine::Input::InputSystem input_system{};
    ApplicationContext context{};
    bool platform_initialized = false;
    bool bootstrap_initialized = false;
    bool runtime_initialized = false;
    bool setup_completed = false;
    // DiscardStale: renders submitted to the queue but not yet presented, in
    // frame order. Presents are issued only once a frame's GPU work completes,
    // so the Mailbox compositor always sees (and scans out) the freshest frames.
    struct PendingPresent { std::uint32_t frame_counter; std::uint32_t image_index; };
    std::vector<PendingPresent> pending_presents{};
    const bool discard_stale = config.present_policy == PresentPolicy::DiscardStale;

    // Wall-clock frame profiling. Off unless the config asks for it, in which
    // case every stage of the loop below is timed and each frame that ran long is
    // charged to the stage that dominated it. This is the only way to see time
    // the loop spends *blocked* -- fences, acquires, presents, or the scheduler.
    Runtime::FrameProfiler frame_profiler{};

    auto cleanup = [&]() {
        VulkanEngine::Crash::Stage("shutdown");
        VulkanShared::Timer t{true};
        double prev = 0.0;
        if (bootstrap_initialized && bootstrap) {
            VulkanShared::ScopedSection drain{
                "shutdown: gpu drain",
                [](const std::string& section, double ms) {
                    LOGIFACE_LOG(debug, section + ": " + std::to_string(ms) + " ms");
                }};
            bootstrap->GetBackend().WaitDeviceIdle();
            prev = t.ElapsedMs();
        }
        if (setup_completed) {
            hooks.on_shutdown.Call(context);
            const double current = t.ElapsedMs();
            LOGIFACE_LOG(debug, "shutdown: app hooks " + std::to_string(current - prev) + " ms");
            prev = current;
            setup_completed = false;
        }
        if (runtime_initialized && runtime) {
            runtime->Shutdown();
            const double current = t.ElapsedMs();
            LOGIFACE_LOG(debug, "shutdown: runtime " + std::to_string(current - prev) + " ms");
            prev = current;
            runtime_initialized = false;
        }
        if (bootstrap_initialized && bootstrap) {
            bootstrap->Shutdown();
            const double current = t.ElapsedMs();
            LOGIFACE_LOG(debug, "shutdown: bootstrap " + std::to_string(current - prev) + " ms");
            prev = current;
            bootstrap_initialized = false;
        }
        if (platform_initialized && platform) {
            platform->Shutdown();
            const double current = t.ElapsedMs();
            LOGIFACE_LOG(debug, "shutdown: platform " + std::to_string(current - prev) + " ms");
            prev = current;
            platform_initialized = false;
        }
        LOGIFACE_LOG(info, "shutdown: total " + std::to_string(t.ElapsedMs()) + " ms");
    };

    auto fail = [&](const std::string& message) -> int {
        LOGIFACE_LOG(error, message);
        cleanup();
        return 1;
    };

    try {
        // Startup timing starts before the logger comes up so the total covers
        // logger and per-user storage initialization too — the counterpart of
        // the per-phase and total timings the shutdown path logs in cleanup().
        const VulkanShared::Timer startup{true};
        double startup_prev = 0.0;
        VulkanEngine::Startup::InitializeLogger(config.log_level);
        LOGIFACE_LOG(info, config.app_name + " started");

        if (config.frame_profile) {
            Runtime::FrameProfileConfig profile_config{};
            profile_config.long_frame_ms = config.frame_profile_long_ms;
            profile_config.log_every = std::max<std::uint64_t>(config.frame_profile_every, 1);
            frame_profiler.Configure(profile_config);
            LOGIFACE_LOG(info, "frame profiling on (long frame >= " +
                                   std::to_string(config.frame_profile_long_ms) +
                                   " ms, summary every " +
                                   std::to_string(profile_config.log_every) + " iterations)");
        }

        // ── Per-user storage ───────────────────────────────────────────────
        // Resolved before the platform and the device, so a saved setting can
        // drive window creation and the shader cache has a real location before
        // the engine bootstraps. Nothing here writes to the executable's
        // directory: an installed application usually cannot.
        VulkanShared::UserPaths::Options storage_options{};
        if (!config.user_dir.empty()) {
            storage_options.base_dir = VulkanShared::UserPaths::FromUtf8(config.user_dir);
        }
        if (!config.executable_path.empty()) {
            storage_options.executable_dir = config.executable_path.parent_path();
        }
        storage_options.force_portable = config.force_portable;
        storage_options.disable_portable = config.disable_portable;

        auto storage_result = VulkanShared::Storage::Storage::Create(
            VulkanShared::UserPaths::Identity{.org = config.org_id, .app = config.app_id},
            storage_options);
        if (!storage_result) {
            return fail("Per-user storage unavailable: " + storage_result.error().ToString());
        }
        storage.emplace(std::move(*storage_result));

        const VulkanShared::UserPaths::Roots& storage_roots = storage->GetRoots();
        LOGIFACE_LOG(info, std::format("user data: {} ({})",
                                       VulkanShared::UserPaths::ToUtf8(storage_roots.persistent),
                                       VulkanShared::UserPaths::ToString(storage_roots.origin)));
        if (!storage_roots.fallback_reason.empty()) {
            LOGIFACE_LOG(warn, storage_roots.fallback_reason);
        }
        // Session logs and crash reports belong with the rest of the user's data
        // rather than beside the executable, which is where Crash defaults to
        // (and where an installed binary is usually not writable).
        //
        // Residual: Crash::Boot runs from the entry point, before argv is parsed,
        // so whatever is logged between then and this point (a couple of lines at
        // most) has already gone to an executable-relative session log. Moving
        // that file is not worth the sequencing; an unwritable directory simply
        // means those few lines are lost, and the crash report still contains the
        // full in-memory ring.
        //
        // Known limitation: Crash.cpp opens these with CreateFileA and derives
        // paths with GetModuleFileNameA, i.e. narrow byte paths. A Windows
        // profile whose name is not representable in the active ANSI codepage
        // therefore loses the session log and crash report (the path is
        // truncated or mangled rather than rejected). Fixing it means switching
        // that file to CreateFileW/GetModuleFileNameW plus an allocation-free
        // UTF-8 conversion on the fault path, which is deliberately not done
        // here: the crash handler is the one subsystem that must not be changed
        // without a Windows runtime to test on.
        VulkanEngine::Crash::SetReportDirectory(
            VulkanShared::UserPaths::ToUtf8(storage_roots.log).c_str());

        VulkanEngine::Crash::Stage("platform init");
        const auto platform_backend = VulkanBackend::Platform::CreateSdlPlatformBackend();
        platform = std::make_unique<VulkanBackend::Platform::SdlPlatform>(platform_backend);

        auto platform_config = config.platform_config;
        if (platform_config.window_title.empty()) {
            platform_config.window_title = config.app_name;
        }
        if (!platform->Initialize(platform_config)) {
            std::string msg = std::string{"Platform initialization failed: "} + std::string{PlatformStatusToString(platform->GetState().status)};
            if (!platform->GetState().error_message.empty()) {
                msg += " (" + platform->GetState().error_message + ")";
            }
            return fail(msg);
        }
        platform_initialized = true;
        {
            const double current = startup.ElapsedMs();
            LOGIFACE_LOG(debug, "startup: platform " + std::to_string(current - startup_prev) + " ms");
            startup_prev = current;
        }

        SDL_Window* window = platform->GetNativeWindowHandle();
        if (window == nullptr) {
            return fail("Native SDL window handle is null");
        }

        vk_backend = VulkanBackend::Vulkan::CreateVulkanBootstrapBackend();
        bootstrap = std::make_unique<VulkanBackend::Vulkan::VulkanBootstrap>(vk_backend);
        auto bootstrap_config = config.bootstrap_config;

        VulkanEngine::Crash::Stage("vulkan bootstrap init");
        // ── Frame pipeline is configured from the single ApplicationConfig ──
        // frame-clock: the device sync rings (command buffers, fences, semaphores)
        // and every engine FIF ring are sized from this value.
        const std::uint32_t fif = std::clamp(config.frames_in_flight, 1u, 4u);
        if (config.frames_in_flight != fif || config.frames_in_flight < 2) {
            LOGIFACE_LOG(warn, "frames_in_flight=" + std::to_string(config.frames_in_flight) +
                         " clamped to " + std::to_string(fif) + "; only 2 or 3 make practical sense");
        }
        if (discard_stale &&
            (fif < 3 ||
             (bootstrap_config.present_mode != VulkanBackend::Vulkan::PresentMode::Mailbox &&
              bootstrap_config.present_mode != VulkanBackend::Vulkan::PresentMode::Immediate))) {
            return fail("PresentPolicy::DiscardStale requires frames_in_flight >= 3 and a "
                        "Mailbox or Immediate present mode (stale-frame discard is performed by the "
                        "swapchain compositor)");
        }
        bootstrap_config.frames_in_flight = fif;

        if (config.swapchain_image_count >= 2) {
            bootstrap_config.preferred_swapchain_image_count = config.swapchain_image_count;
        } else {
            // Derive from present mode: with FIFO the present queue holds an image
            // until its VBlank turn, so it cannot be re-acquired meanwhile; Mailbox
            // drops pending presents, so images become free as soon as the render
            // completes.
            const bool fifo_present =
                bootstrap_config.present_mode == VulkanBackend::Vulkan::PresentMode::Fifo ||
                bootstrap_config.present_mode == VulkanBackend::Vulkan::PresentMode::FifoRelaxed;
            bootstrap_config.preferred_swapchain_image_count =
                fifo_present ? fif + 1u : std::max(fif, 2u);
        }
        if (discard_stale) {
            // Deferred presents can hold up to FIF-1 images while the GPU catches
            // up; budget one spare so acquire never stalls on image reuse.
            bootstrap_config.preferred_swapchain_image_count =
                std::max(bootstrap_config.preferred_swapchain_image_count, fif + 1u);
        }
        bootstrap_config.native_window_handle = window;
        if (!bootstrap->Initialize(bootstrap_config)) {
            std::string bootstrap_message = "Vulkan bootstrap initialization failed";
            if (!bootstrap->GetSnapshot().error_message.empty()) {
                bootstrap_message += ": " + bootstrap->GetSnapshot().error_message;
            }
            return fail(bootstrap_message);
        }
        bootstrap_initialized = true;
        {
            const double current = startup.ElapsedMs();
            LOGIFACE_LOG(debug, "startup: vulkan bootstrap " + std::to_string(current - startup_prev) + " ms");
            startup_prev = current;
        }

        VulkanEngine::Crash::Stage("runtime init");
        runtime = std::make_unique<VulkanBackend::Vulkan::FrameLoop>();
        if (!runtime->Initialize(config.runtime_config)) {
            return fail("Runtime shell initialization failed");
        }
        runtime_initialized = true;
        {
            const double current = startup.ElapsedMs();
            LOGIFACE_LOG(debug, "startup: runtime " + std::to_string(current - startup_prev) + " ms");
            startup_prev = current;
        }

        context.platform = platform.get();
        context.runtime = runtime.get();
        context.bootstrap = bootstrap.get();
        context.input_system = &input_system;
        context.storage = &*storage;
        context.window = window;
        context.platform_state = &platform->GetState();
        context.geometry_buffer_size_mb = config.geometry_buffer_size_mb;
        context.draw_mode = config.draw_mode;
        context.system_profile_sink = frame_profiler.Enabled() ? &frame_profiler : nullptr;
        context.gpu_pass_timing = config.gpu_pass_timing;
        context.gpu_pass_timing_every = config.gpu_pass_timing_every;

        VulkanEngine::Crash::Stage("app setup");
        if (!hooks.on_setup.Call(context)) {
            return fail("Application setup failed");
        }
        setup_completed = true;
        {
            const double current = startup.ElapsedMs();
            LOGIFACE_LOG(debug, "startup: app setup " + std::to_string(current - startup_prev) + " ms");
            startup_prev = current;
        }

        Runtime::FrameStepClock step_clock{};
        step_clock.Configure(config.fixed_timestep, config.max_fixed_steps_per_frame,
                             config.max_frame_delta);
        bool step_budget_logged = false;
        auto previous_time = std::chrono::steady_clock::now();
        std::uint32_t rendered_frames = 0;

        // Startup total at the frame-loop boundary. It includes the game's
        // app setup; the engine-only figure is this minus the "startup:
        // app setup" phase above. Runs after it, everything below is the
        // steady state.
        LOGIFACE_LOG(info, "startup: total " + std::to_string(startup.ElapsedMs()) + " ms");
        VulkanEngine::Crash::Stage("frame loop");
        while (!platform->ShouldQuit() && !runtime->ShouldShutdown()) {
            // First statement of the iteration: closes the interval that just
            // ended, so it covers poll, input, acquire and present too.
            frame_profiler.BeginIteration();

            // Sampled once at the top of the iteration, ahead of every path that can
            // `continue` (minimized, swapchain recreation, acquire failure). Those
            // paths used to return before the sample, so the entire stall they
            // absorbed was banked into the next frame's delta. Advancing the clock
            // unconditionally here makes each of them contribute one ordinary,
            // clamped delta instead. delta_time therefore measures iteration-start to
            // iteration-start — the full frame period, including the current
            // iteration's acquire and present.
            const auto now = std::chrono::steady_clock::now();
            const float raw_delta_time = std::chrono::duration<float>(now - previous_time).count();
            previous_time = now;

            VulkanBackend::Event::EventList platform_events;
            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::PollEvents};
                platform_events = platform->PollEvents();
            }

            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::PreInput};
                hooks.on_pre_input.Call(context);
            }

            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::InputProcess};
                bool filter_mouse = hooks.should_filter_mouse_input && hooks.should_filter_mouse_input();
                bool filter_keyboard = hooks.should_filter_keyboard_input && hooks.should_filter_keyboard_input();

                VulkanBackend::Event::EventList filtered_events;
                filtered_events.reserve(platform_events.size());

                for (auto& event : platform_events) {
                    const auto etype = event->GetEventType();
                    bool is_mouse = etype == VulkanBackend::Event::EventType::MouseButtonDown ||
                                    etype == VulkanBackend::Event::EventType::MouseButtonUp ||
                                    etype == VulkanBackend::Event::EventType::MouseMotion ||
                                    etype == VulkanBackend::Event::EventType::MouseWheel;

                    bool is_keyboard = etype == VulkanBackend::Event::EventType::KeyDown ||
                                       etype == VulkanBackend::Event::EventType::KeyUp;

                    if (is_mouse && filter_mouse) {
                        continue;
                    }
                    if (is_keyboard && filter_keyboard) {
                        continue;
                    }
                    filtered_events.push_back(std::move(event));
                }

                input_system.ProcessEvents(filtered_events);
                if (context.quit_action_handle.id != UINT32_MAX && input_system.WasActionStarted(context.quit_action_handle)) {
                    runtime->RequestShutdown();
                }

                context.platform_state = &platform->GetState();
                if (context.platform_state->quit_requested) {
                    runtime->RequestShutdown();
                }

                runtime->NotifyWindowMinimized(context.platform_state->minimized);
            }

            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::RuntimeBegin};
                context.frame.runtime_frame = runtime->BeginFrame();
            }
            if (context.frame.runtime_frame.status == VulkanBackend::Vulkan::RuntimeStatus::Minimized ||
                context.frame.runtime_frame.status == VulkanBackend::Vulkan::RuntimeStatus::ShutdownRequested) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config.minimized_sleep_ms));
                continue;
            }

            if (context.platform_state->resized) {
                bootstrap->NotifySwapchainOutOfDate();
            }

            // DiscardStale: release frames whose GPU work has completed. Presents
            // are issued in submission order; the Mailbox compositor then scans out
            // the freshest completed frame and drops the stale ones.
            if (discard_stale) {
                // Deferred presents and their fence checks happen here, not in
                // the sequential Present scope; name the stage explicitly so a
                // blocked present does not disappear into `unaccounted`.
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::PresentDrain};
                while (!pending_presents.empty()) {
                    PendingPresent& front = pending_presents.front();
                    // The front frame's slot fence reflects exactly its own work: no
                    // later frame has reused the slot (that reuse is gated on this
                    // same fence by AcquireNextImage), so a signaled fence means the
                    // frame itself is GPU-complete.
                    if (!bootstrap->IsFrameComplete(front.frame_counter)) {
                        break;
                    }
                    if (!bootstrap->Present(front.image_index)) {
                        bootstrap->NotifySwapchainOutOfDate();
                        break;
                    }
                    frame_profiler.NotePresent();
                    pending_presents.erase(pending_presents.begin());
                }
            }

            VulkanBackend::Vulkan::VulkanBootstrapState bootstrap_frame{};
            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::BootstrapBegin};
                bootstrap_frame = bootstrap->BeginFrame();
            }
            if (bootstrap_frame.status == VulkanBackend::Vulkan::BootstrapStatus::SwapchainOutOfDate) {
                bool recreated = false;
                {
                    const Runtime::FrameStageScope stage_scope{
                        frame_profiler, Runtime::FrameStage::SwapchainRecreate};
                    recreated = bootstrap->RecreateSwapchain();
                }
                if (!recreated) {
                    return fail("Swapchain recreation failed");
                }
                // Recreation is unbounded work performed after this iteration's
                // sample; re-stamp the clock so it is billed to no frame at all.
                // The profiler must agree, or the one-off stall lands in the next
                // interval and is blamed on whatever happens then.
                previous_time = std::chrono::steady_clock::now();
                frame_profiler.DiscardNextInterval();
                continue;
            }
            if (bootstrap_frame.status != VulkanBackend::Vulkan::BootstrapStatus::Ok) {
                return fail("Vulkan bootstrap entered non-OK frame status");
            }

            context.frame.image_index = 0;
            bool acquired = false;
            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::Acquire};
                acquired = bootstrap->AcquireNextImage(context.frame.image_index);
            }
            if (!acquired) {
                bootstrap->NotifySwapchainOutOfDate();
                continue;
            }

            const Runtime::FrameStep tick = step_clock.Advance(raw_delta_time);
            context.frame.delta_time = tick.delta_time;
            context.frame.fixed_delta_time = tick.step_delta;
            context.frame.fixed_step_count = tick.step_count;
            context.frame.interpolation_alpha = tick.alpha;

            if (tick.budget_exhausted && !step_budget_logged) {
                step_budget_logged = true;
                LOGIFACE_LOG(warn, "fixed-step budget exhausted (" +
                                       std::to_string(config.max_fixed_steps_per_frame) +
                                       " steps in one iteration); dropping the backlog. The frame "
                                       "rate is too low for fixed_timestep=" +
                                       std::to_string(config.fixed_timestep) + "s");
            }

            context.frame.frame_counter = bootstrap->GetSnapshot().frame_index;

            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::AppUpdate};
                for (std::uint32_t step = 0; step < tick.step_count; ++step) {
                    hooks.on_fixed_update.Call(context);
                    if (runtime->ShouldShutdown()) {
                        // Do not simulate further once a step has asked to quit.
                        break;
                    }
                }

                hooks.on_frame_update.Call(context);
            }

            context.frame.render_success = true;
            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::AppRender};
                hooks.on_frame_render.Call(context);
            }

            if (discard_stale) {
                // Submit now, present later (once the GPU work completes — see the
                // drain at the top of the loop). Frames whose present is superseded
                // are dropped by the Mailbox compositor; every image is still
                // presented exactly once, so none leaks.
                bool submitted = false;
                {
                    const Runtime::FrameStageScope stage_scope{
                        frame_profiler, Runtime::FrameStage::Submit};
                    submitted = bootstrap->SubmitFrame(context.frame.image_index,
                                                       context.frame.render_success);
                }
                if (!submitted) {
                    bootstrap->NotifySwapchainOutOfDate();
                } else if (context.frame.render_success) {
                    pending_presents.push_back(
                        {context.frame.frame_counter, context.frame.image_index});
                }
            } else {
                bool submitted = false;
                {
                    const Runtime::FrameStageScope stage_scope{
                        frame_profiler, Runtime::FrameStage::Submit};
                    submitted = bootstrap->SubmitFrame(context.frame.image_index,
                                                       context.frame.render_success);
                }
                if (!submitted) {
                    bootstrap->NotifySwapchainOutOfDate();
                } else if (context.frame.render_success) {
                    bool presented = false;
                    {
                        const Runtime::FrameStageScope stage_scope{
                            frame_profiler, Runtime::FrameStage::Present};
                        presented = bootstrap->Present(context.frame.image_index);
                    }
                    if (presented) {
                        frame_profiler.NotePresent();
                    } else {
                        bootstrap->NotifySwapchainOutOfDate();
                    }
                }
            }

            ++rendered_frames;
            if (rendered_frames == 1) {
                // Light game-side marker: time from process entry (well, logger
                // init) to the first presented frame. Covers pipeline/shader
                // first-use and the first acquire that the pre-loop total hides.
                LOGIFACE_LOG(info, "startup: first frame " + std::to_string(startup.ElapsedMs()) + " ms");
            }
            if (config.max_frames != 0 && rendered_frames >= config.max_frames) {
                runtime->RequestShutdown();
            }

            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::RuntimeEnd};
                runtime->EndFrame();
            }
            {
                const Runtime::FrameStageScope stage_scope{
                    frame_profiler, Runtime::FrameStage::BootstrapEnd};
                bootstrap->EndFrame();
            }

            frame_profiler.EndIteration();
            if (frame_profiler.TakeSummaryDue()) {
                // The summary write is synchronous and slow enough to look like a
                // stutter of its own, so the interval that contains it is dropped
                // rather than measured: the instrument does not appear in its own
                // numbers.
                frame_profiler.DiscardNextInterval();
                LOGIFACE_LOG(info, frame_profiler.Summary());
                // Start the next window clean, so consecutive summaries describe
                // disjoint spans rather than re-reporting the whole run.
                frame_profiler.ResetWindow();
            }
        }

        // A run shorter than the summary cadence still gets its window: without
        // this, `--frame-profile --max-frames 600` (the default cadence) would
        // print only the banner. HasSamples() covers a one-iteration run too,
        // which records stage time but no closed interval.
        if (frame_profiler.Enabled() && frame_profiler.HasSamples()) {
            LOGIFACE_LOG(info, frame_profiler.Summary());
        }

        cleanup();
        LOGIFACE_LOG(info, "App completed");
        return 0;
    } catch (const std::exception& ex) {
        LOGIFACE_LOG(error, std::string("Fatal error: ") + ex.what());
        cleanup();
        return 1;
    }
}
// NOLINTEND

} // namespace VulkanEngine::Application
