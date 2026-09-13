# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Kit entry point for the Synaptics tactile sensor extension."""

import carb
import omni.ext
import omni.kit.app
import omni.kit.menu.utils as menu_utils
import omni.timeline
import omni.usd
from omni.kit.menu.utils import MenuItemDescription
from pxr import UsdPhysics

from .adapters import get_newton_adapter
from .config import available_models, ensure_core_package_on_path, resolve_extension_root
from .diagnostics import format_diagnostics, run_diagnostics
from .runtime import TactileRuntime, set_active_runtime
from .scenario import (
    DEFAULT_SCENE,
    DEFAULT_TIME_CODES_PER_SECOND,
    PHYSICS_RATE_HZ,
    build_scenario,
    reset_simulation,
)
from .spawn import spawn_sensor
from .ui_builder import WINDOW_TITLE, TactileSensorWindow

#: Submenu placement under Create.
CREATE_MENU_ROOT = "Sensors"
SENSOR_MENU_NAME = "Synaptics Tactile Sensor"

_LOG_PREFIX = "[Synaptics Tactile]"


class SynapticsTactileSensorExtension(omni.ext.IExt):
    """Registers the menus and the panel; owns nothing else.

    Sensor math lives in the ``synaptics_tactile_newton`` package and the
    Isaac Sim binding lives behind ``adapters/`` — this class stays a shell so
    that a future change to Kit's extension model is a small re-port.
    """

    def on_startup(self, extension_id: str) -> None:
        # A repo checkout carries the core package beside isaacsim_ext/; make it
        # importable before anything lazily imports CTSSensor.
        ensure_core_package_on_path()

        self._extension_root = None
        self._menu_items = []
        self._window_menu_items = []
        self._window = None
        self._one_frame_sub = None

        try:
            self._extension_root = resolve_extension_root(extension_id)
        except Exception as error:  # noqa: BLE001 - fall back to the on-disk layout.
            carb.log_warn(f"{_LOG_PREFIX} Could not resolve extension path: {error}")

        self._models = available_models(self._extension_root)
        if not self._models:
            carb.log_warn(f"{_LOG_PREFIX} No sensor model profiles found in data/models.")

        # Binds CTSSensor to the live Newton model on Play, tears down on Stop.
        self._runtime = TactileRuntime()
        self._runtime.start()
        set_active_runtime(self._runtime)

        self._window = TactileSensorWindow(
            models=self._models,
            on_spawn=self._spawn_sensor,
            on_diagnostics=self._diagnostics_text,
            on_load_scenario=self._load_scenario,
            on_reset_scenario=self._reset_scenario,
            on_read_forces=self._runtime_readout,
            on_warm_up=self._warm_up_kernels,
            on_step=self._step_one_frame,
            on_play_one=self._play_one_frame,
        )

        self._menu_items = [
            MenuItemDescription(
                name=CREATE_MENU_ROOT,
                sub_menu=[
                    MenuItemDescription(
                        name=SENSOR_MENU_NAME,
                        sub_menu=[
                            MenuItemDescription(
                                name=model_name,
                                onclick_fn=(
                                    lambda selected=model_name: self._spawn_sensor(selected)
                                ),
                            )
                            for model_name in self._models
                        ],
                    )
                ],
            )
        ]
        menu_utils.add_menu_items(self._menu_items, "Create")

        self._window_menu_items = [
            MenuItemDescription(name=WINDOW_TITLE, onclick_fn=self._toggle_window)
        ]
        menu_utils.add_menu_items(self._window_menu_items, "Window")
        menu_utils.rebuild_menus()

        carb.log_info(f"{_LOG_PREFIX} Extension started ({extension_id}).")

    def on_shutdown(self) -> None:
        """Tear down so nothing holds a reference to this object.

        Kit warns ("extension object is still alive, something holds a reference
        on it") when an ``IExt`` survives shutdown. Everything we hand out — the
        panel's five callbacks and the menu items' click handlers — is a bound
        method or a closure over ``self``, so each has to be dropped explicitly;
        dropping our own reference to the *container* is not enough, because Kit
        and ``omni.ui`` keep those objects alive a while longer.
        """
        self._one_frame_sub = None

        if getattr(self, "_runtime", None) is not None:
            self._runtime.stop()
            self._runtime = None
        set_active_runtime(None)

        if self._menu_items:
            menu_utils.remove_menu_items(self._menu_items, "Create")
            self._release_menu_callbacks(self._menu_items)
            self._menu_items = []
        if self._window_menu_items:
            menu_utils.remove_menu_items(self._window_menu_items, "Window")
            self._release_menu_callbacks(self._window_menu_items)
            self._window_menu_items = []
        menu_utils.rebuild_menus()

        if self._window is not None:
            self._window.destroy()
            self._window = None

        carb.log_info(f"{_LOG_PREFIX} Extension shut down.")

    @classmethod
    def _release_menu_callbacks(cls, items) -> None:
        """Drop ``onclick_fn`` from a menu tree so its closures release ``self``."""
        for item in items or ():
            if getattr(item, "onclick_fn", None) is not None:
                item.onclick_fn = None
            cls._release_menu_callbacks(getattr(item, "sub_menu", None))

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _toggle_window(self) -> None:
        if self._window is not None:
            self._window.toggle()

    def _spawn_sensor(self, model_name: str) -> str:
        """Spawn ``model_name`` under the current selection. Returns a status line."""
        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            message = "No stage is open."
            carb.log_error(f"{_LOG_PREFIX} {message}")
            return message

        selection = context.get_selection().get_selected_prim_paths()
        parent_path = selection[0] if selection else "/World"

        try:
            sensor_path, config = spawn_sensor(
                stage,
                model_name,
                parent_path=parent_path,
                extension_root=self._extension_root,
            )
        except (FileNotFoundError, ValueError) as error:
            carb.log_error(f"{_LOG_PREFIX} {error}")
            return str(error)

        message = (
            f"Spawned {model_name} at {sensor_path} "
            f"({config['numTaxels']} taxels, under {parent_path})."
        )
        carb.log_info(f"{_LOG_PREFIX} {message}")
        return message

    def _load_scenario(self, model_name: str, scene: str = DEFAULT_SCENE) -> str:
        """Build the whole demo scene in one action. Returns a status line."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return "No stage is open."

        # Rebuilding the stage under a running simulation forces Newton to
        # re-initialize mid-play, which is a needless way to get a half-torn-down
        # model. Stop first; the user presses Play when the scene is ready.
        omni.timeline.get_timeline_interface().stop()

        try:
            info = build_scenario(
                stage, model_name, extension_root=self._extension_root, scene=scene
            )
        except (FileNotFoundError, ValueError) as error:
            carb.log_error(f"{_LOG_PREFIX} {error}")
            return str(error)

        count = len(info["indenter_paths"])
        lines = [
            f"Loaded {model_name} / {info['scene_label']}: {info['num_taxels']} taxels, "
            f"{count} indenter{'' if count == 1 else 's'} totalling "
            f"{info['indenter_mass_kg'] * 1000:.0f} g "
            f"(weight ~{info['expected_total_force_n']:.3f} N). Press Play."
        ]
        slow_motion = DEFAULT_TIME_CODES_PER_SECOND / info["time_codes_per_second"]
        if slow_motion < 1.0:
            lines.append(
                f"Playback is {1.0 / slow_motion:.0f}x slow motion "
                f"({info['time_codes_per_second']:.0f} fps of sim time) — the drop "
                "is over in a tenth of a second at real speed."
            )
        if info["overhanging"]:
            lines.append(
                f"Warning: {', '.join(info['overhanging'])} overhangs the taxel array, "
                "so the coplanar base will carry most of the load."
            )
        lines.append(
            "First Play of a session pauses while Warp compiles the MuJoCo contact "
            "kernels (~30 s on Isaac Sim 6.0.1, several minutes on 6.1.0) — it is "
            "not hung, and the compile is cached after that."
        )
        message = "\n".join(lines)
        carb.log_info(f"{_LOG_PREFIX} {message}")
        return message

    def _reset_scenario(self) -> str:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return "No stage is open."
        if reset_simulation(stage):
            return "Reset: timeline stopped, indenters back at their drop heights. Press Play."
        return "No scenario on this stage — press Load Scenario first."

    def _runtime_readout(self) -> dict:
        """Current readout for the panel: text plus per-sensor arrays."""
        if self._runtime is None:
            return {"text": "Runtime is not started.", "sensors": [], "error": ""}
        try:
            return self._runtime.readout()
        except Exception as error:  # noqa: BLE001 - the panel must never take the app down.
            carb.log_error(f"{_LOG_PREFIX} force readout failed: {error}")
            return {
                "text": f"Force readout failed: {type(error).__name__}: {error}",
                "sensors": [],
                "error": str(error),
            }

    @staticmethod
    def _physics_rate_hz(stage) -> float:
        """Physics steps per second, usable *before* the first step.

        Once Newton has a model its ``sim_dt`` is authoritative. Before that it
        still reports something — its 600 Hz config default — which is not what
        the scene asked for, so fall back to the rate authored on the physics
        scene instead of trusting it.
        """
        try:
            adapter = get_newton_adapter()
        except ImportError:
            adapter = None
        if adapter is not None and adapter.is_ready():
            physics_dt = adapter.physics_dt()
            if physics_dt:
                return 1.0 / physics_dt

        if stage is not None:
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    attribute = prim.GetAttribute("physxScene:timeStepsPerSecond")
                    value = attribute.Get() if attribute else None
                    if value:
                        return float(value)
        return PHYSICS_RATE_HZ

    def _play_one_frame(self) -> str:
        """Start the simulation and stop after a single **physics step**.

        The timeline has no "play one step" control: Play runs until you catch
        it, and Step needs a *paused* timeline, which from a stopped stage means
        Play-then-Pause by hand. This does that pair for you and leaves the
        simulation paused and ready to Step.

        Sets the timeline rate to the physics rate first and defers the pause to
        a one-shot update subscription — see :meth:`_step_one_frame` for why
        both are necessary.
        """
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            timeline.pause()
            return "Paused. Use Step 1 / Step 5 to advance."

        stage = omni.usd.get_context().get_stage()
        rate = self._physics_rate_hz(stage)
        if stage is not None and rate:
            stage.SetTimeCodesPerSecond(rate)

        timeline.play()
        self._one_frame_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._pause_after_frame, name="synaptics.tactile.oneframe")
        )
        return (
            f"Playing one physics step ({1000.0 / rate:.2f} ms), then pausing — "
            f"Step 1 / Step 5 from here. (The first step of a session also "
            f"compiles the physics kernels unless you warmed them up.)"
        )

    def _pause_after_frame(self, event) -> None:
        self._one_frame_sub = None  # one-shot
        omni.timeline.get_timeline_interface().pause()

    def _step_one_frame(self, physics_steps: int = 1) -> str:
        """Advance the simulation by exactly ``physics_steps`` physics steps.

        Isaac Sim has no step-one-physics-frame control in the Newton app: the
        Animation Timeline moves the *playhead*, and per omni.kit.loop-isaac's
        docs "timeline tick advance and physics stepping are separate systems".
        ``forward_one_frame()`` does advance physics, but one frame is
        ``1/timeCodesPerSecond`` of sim time — 17 physics steps at the default
        60 fps against a 1 kHz solver, which is a jump, not a step.

        So the timeline rate is retuned to ``1/(physics_steps * physics_dt)``
        first, making one frame exactly the requested number of steps. That also
        leaves playback in slow motion, which is usually what someone stepping
        wants; Reset puts it back to real time.

        It never pumps ``app.update()`` — doing that from a UI callback
        re-enters the frame loop and crashes Kit.
        """
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_stopped():
            return "Nothing to step — press Play 1 first to start the simulation and pause it."
        if timeline.is_playing():
            timeline.pause()

        stage = omni.usd.get_context().get_stage()
        try:
            physics_dt = get_newton_adapter().physics_dt()
        except ImportError as error:
            return str(error)
        if stage is not None and physics_dt:
            rate = 1.0 / (max(1, int(physics_steps)) * physics_dt)
            if abs(stage.GetTimeCodesPerSecond() - rate) > 1e-6:
                stage.SetTimeCodesPerSecond(rate)

        timeline.forward_one_frame()

        if not physics_dt:
            return "Stepped one frame."
        return (
            f"Stepped {physics_steps} physics step"
            f"{'' if physics_steps == 1 else 's'} "
            f"({physics_steps * physics_dt * 1000:.2f} ms). Playback is slowed to "
            f"match; Reset restores real time."
        )

    def _warm_up_kernels(self) -> str:
        """Compile the physics kernels now, by stepping *physics* directly.

        The expensive kernels are generated from the finalized model, so they
        cannot be force-loaded ahead of a first step — something has to step.

        It steps the Newton backend, **not the app**: driving the timeline and
        pumping ``app.update()`` from a UI callback re-enters the frame loop and
        corrupts the renderer's command buffer, which hard-crashes Kit. Calling
        the physics interface touches no frame and no timeline.
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return "No stage is open."

        try:
            adapter = get_newton_adapter()
        except ImportError as error:
            return str(error)

        interface = adapter.physics_interface()
        newton_stage = adapter.newton_stage()
        if interface is None or newton_stage is None:
            return "Newton backend is not ready — is the Newton experience running?"

        try:
            # Building the model compiles the collision kernels — and sets
            # sim_dt from the scene, so the step below uses the real rate.
            interface.force_load_physics_from_usd()
            # ...and one solver step compiles MuJoCo's constraint kernels.
            newton_stage.step_sim(newton_stage.sim_dt)
        except Exception as error:  # noqa: BLE001 - report, never take the app down.
            carb.log_error(f"{_LOG_PREFIX} warm-up failed: {error}")
            return f"Warm-up failed: {type(error).__name__}: {error}"

        return (
            "Physics kernels compiled and cached — Play should be immediate now. "
            "The cache lives in ~/.cache/warp, so this is one-time per machine."
        )

    def _diagnostics_text(self, verbose: bool = False) -> str:
        try:
            return format_diagnostics(run_diagnostics(), verbose=verbose)
        except Exception as error:  # noqa: BLE001 - the panel must never take the app down.
            carb.log_error(f"{_LOG_PREFIX} Diagnostics failed: {error}")
            return f"Diagnostics failed: {type(error).__name__}: {error}"
