# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""The extension's ``omni.ui`` panel.

Holds no sensor state: it pulls a readout dict from the runtime and renders it.
The heatmap lives in :mod:`taxel_grid`, its geometry in :mod:`taxel_layout`.

The grid refreshes on an app-update subscription rather than a button, throttled
to ~10 Hz.
"""

import carb
import omni.kit.app
import omni.ui as ui

from .config import DEFAULT_FORCE_MAX_N
from .scenario import SCENES, scene_keys, scene_labels
from .taxel_grid import TaxelGrid

WINDOW_TITLE = "Synaptics Tactile Sensor"

_HEADER_STYLE = {"font_size": 18}
_REPORT_STYLE = {"font_size": 13}

#: Refresh the heatmap every Nth app update (~10 Hz at 60 fps).
_REFRESH_EVERY = 6


class TactileSensorWindow:
    """The extension's single panel."""

    def __init__(
        self,
        models,
        on_spawn,
        on_diagnostics,
        on_load_scenario,
        on_reset_scenario,
        on_read_forces,
        on_warm_up,
        on_step,
        on_play_one,
    ):
        """
        Args:
            models: Model profile names to offer, e.g. ``["CTS0.0"]``.
            on_spawn: ``callable(model_name) -> str`` returning a status line.
            on_diagnostics: ``callable(verbose: bool) -> str`` returning the
                formatted preflight report.
            on_load_scenario: ``callable(model_name, scene_key) -> str`` building
                the demo scene.
            on_reset_scenario: ``callable() -> str`` resetting the scene.
            on_read_forces: ``callable() -> dict`` — the runtime readout (see
                ``TactileRuntime.readout``).
            on_warm_up: ``callable() -> str`` pre-compiling the physics kernels.
            on_step: ``callable(physics_steps: int) -> str`` advancing the sim.
            on_play_one: ``callable() -> str`` playing a single frame.
        """
        self._models = list(models) or ["(no model profiles found)"]
        self._on_spawn = on_spawn
        self._on_diagnostics = on_diagnostics
        self._on_load_scenario = on_load_scenario
        self._on_reset_scenario = on_reset_scenario
        self._on_read_forces = on_read_forces
        self._on_warm_up = on_warm_up
        self._on_step = on_step
        self._on_play_one = on_play_one

        self._report_text = "Press 'Run diagnostics' to check this environment."
        self._status_text = ""
        self._headline_text = "No sensor bound."
        self._model_combo = None
        self._scene_combo = None
        self._scene_hint_label = None
        self._report_label = None
        self._status_label = None
        self._headline_label = None

        self._scene_keys = scene_keys()
        self._grid = TaxelGrid()
        self._grid_frame = None
        self._pending_centroids = None
        self._pending_names = None
        self._frame_counter = 0
        self._warm_up_sub = None

        self._window = ui.Window(WINDOW_TITLE, width=640, height=820, visible=False)
        self._window.frame.set_build_fn(self._build_frame)

        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_app_update, name="synaptics.tactile.ui")
        )

    # ------------------------------------------------------------------ #
    # Window lifecycle
    # ------------------------------------------------------------------ #

    @property
    def visible(self) -> bool:
        return bool(self._window and self._window.visible)

    def toggle(self) -> None:
        if self._window is None:
            return
        self._window.visible = not self._window.visible

    def destroy(self) -> None:
        """Release the window, the subscriptions and every callback.

        The callbacks are bound methods of the extension object — see
        ``SynapticsTactileSensorExtension.on_shutdown`` for why each has to be
        dropped explicitly.
        """
        self._update_sub = None
        self._warm_up_sub = None
        if self._window is not None:
            self._window.frame.set_build_fn(None)
            self._window.destroy()
            self._window = None
        self._grid.destroy()
        if self._grid_frame is not None:
            self._grid_frame.set_build_fn(None)
            self._grid_frame = None
        self._model_combo = None
        self._scene_combo = None
        self._scene_hint_label = None
        self._report_label = None
        self._status_label = None
        self._headline_label = None
        self._on_spawn = None
        self._on_diagnostics = None
        self._on_load_scenario = None
        self._on_reset_scenario = None
        self._on_read_forces = None
        self._on_warm_up = None
        self._on_step = None
        self._on_play_one = None

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build_frame(self) -> None:
        """Build the panel.

        Everything lives inside a ScrollingFrame: the panel is tall, and a
        short window would otherwise put the diagnostics buttons past the
        bottom edge with no way to reach them but resizing the window.

        One scrollbar, not two — the report is a plain label inside this same
        frame rather than its own scroller, so the wheel never lands on a
        nested scroll region and stops.
        """
        with ui.ScrollingFrame():
            with ui.VStack(spacing=8, height=0):
                ui.Label("Synaptics Capacitive Tactile Sensor", style=_HEADER_STYLE, height=24)
                ui.Label(
                    "Newton-backend tactile sensor. Requires the Newton physics "
                    "engine (Isaac Sim 6.0+) with the MuJoCo GPU solver.",
                    word_wrap=True,
                    height=0,
                )

                ui.Spacer(height=6)
                with ui.HStack(height=26, spacing=6):
                    ui.Label("Model", width=60)
                    self._model_combo = ui.ComboBox(0, *self._models)
                # One Load button for N scenes, rather than a button per scene:
                # the list is expected to grow.
                with ui.HStack(height=26, spacing=6):
                    ui.Label("Scene", width=60)
                    self._scene_combo = ui.ComboBox(0, *scene_labels())
                    self._scene_combo.model.add_item_changed_fn(self._scene_changed)
                self._scene_hint_label = ui.Label(
                    SCENES[self._scene_keys[0]].description, word_wrap=True, height=0
                )
                # Scene setup, then transport. Transport reads left to right in the
                # order you use it: start, step, step further, start over.
                with ui.HStack(height=30, spacing=6):
                    ui.Button(
                        "Load Scenario",
                        clicked_fn=self._load_scenario_clicked,
                        tooltip="Build the complete demo scene for the selected "
                                "Scene: ground, light, the sensor mounted pads-up, "
                                "the indenters above it, and a camera that can "
                                "actually see a 4 mm part.",
                    )
                    ui.Button(
                        "Warm up kernels",
                        clicked_fn=self._warm_up_clicked,
                        tooltip="Compile Warp's MuJoCo contact kernels now, instead "
                                "of stalling ~30 s on your first Play.",
                    )
                with ui.HStack(height=26, spacing=6):
                    ui.Button(
                        "Add sensor only",
                        clicked_fn=self._spawn_clicked,
                        tooltip="Reference the sensor asset under the selected prim, "
                                "with no scene around it. Newton builds no model at "
                                "all from a stage with no dynamic body.",
                    )
                with ui.HStack(height=30, spacing=6):
                    ui.Button(
                        "Play 1",
                        clicked_fn=self._play_one_clicked,
                        tooltip="Play a single physics step and pause, so the "
                                "simulation is running and ready to Step. The "
                                "timeline has no play-one-step control of its own.",
                    )
                    ui.Button(
                        "Step 1",
                        clicked_fn=lambda: self._step_clicked(1),
                        tooltip="Advance exactly one physics step while paused. "
                                "Retunes the timeline rate to make that possible, "
                                "so playback also slows; Reset restores real time.",
                    )
                    ui.Button(
                        "Step 5",
                        clicked_fn=lambda: self._step_clicked(5),
                        tooltip="Advance five physics steps while paused.",
                    )
                    ui.Button(
                        "Reset",
                        clicked_fn=self._reset_scenario_clicked,
                        tooltip="Stop the timeline, lift the indenter back to its "
                                "drop height and restore real-time playback. "
                                "Stopping is what rewinds the scene.",
                    )
                self._status_label = ui.Label(self._status_text, word_wrap=True, height=0)

                ui.Spacer(height=10)
                ui.Label("Per-taxel force", style=_HEADER_STYLE, height=22)
                self._headline_label = ui.Label(self._headline_text, word_wrap=True, height=0)
                self._grid_frame = ui.Frame(height=0)
                self._grid_frame.set_build_fn(self._build_grid)

                ui.Spacer(height=10)
                with ui.HStack(height=30, spacing=6):
                    ui.Button("Run diagnostics", clicked_fn=lambda: self._diagnostics_clicked(False))
                    ui.Button("Run diagnostics (verbose)", clicked_fn=lambda: self._diagnostics_clicked(True))
                # word_wrap keeps long recommendation lines readable; without it
                # they run off the right edge behind a horizontal scrollbar.
                self._report_label = ui.Label(
                    self._report_text,
                    style=_REPORT_STYLE,
                    alignment=ui.Alignment.LEFT_TOP,
                    word_wrap=True,
                    height=0,
                )

    def _build_grid(self) -> None:
        if self._pending_centroids is None:
            ui.Label("Load a scenario and press Play.", height=0)
            return
        self._grid.build(self._pending_centroids, self._pending_names)

    # ------------------------------------------------------------------ #
    # Live refresh
    # ------------------------------------------------------------------ #

    def _on_app_update(self, event) -> None:
        if not self.visible or self._on_read_forces is None:
            return
        self._frame_counter += 1
        if self._frame_counter % _REFRESH_EVERY:
            return
        try:
            self._grid.sync_cell_height()
            self._apply_readout(self._on_read_forces())
        except Exception as error:  # noqa: BLE001 - a refresh must not kill the app.
            # Stop refreshing rather than raise every frame, but say so: a silent
            # freeze looks exactly like a sensor reading nothing.
            self._update_sub = None
            carb.log_error(f"[Synaptics Tactile] live refresh stopped: {error!r}")
            self._set_headline(f"Live refresh stopped: {type(error).__name__}: {error}")

    def _apply_readout(self, readout: dict) -> None:
        sensors = readout.get("sensors") or []
        if not sensors:
            self._set_headline(readout.get("error") or "No sensor bound.")
            self._grid.clear()
            return

        sensor = sensors[0]
        centroids = sensor.get("centroids")
        forces = sensor.get("forces")
        if centroids is None or forces is None:
            self._set_headline("Sensor bound, but it exposes no taxel geometry.")
            self._grid.clear()
            return

        if not self._grid.matches(centroids):
            # Frame rebuilds are deferred to the next draw, so this refresh only
            # schedules the layout; the following one paints it.
            self._pending_centroids = centroids
            self._pending_names = sensor.get("names")
            if self._grid_frame is not None:
                self._grid_frame.rebuild()
            return

        stats = self._grid.update(
            forces,
            sensor.get("force_max", DEFAULT_FORCE_MAX_N),
            fixed_scale=self._scene_force_scale(),
        )
        others = len(sensors) - 1
        mode = "Winkler" if sensor.get("winkler") else "raw"
        self._set_headline(
            f"{sensor['prim_path']}   |   {mode} readout   |   "
            f"Σ {float(forces.sum()):.4f} N over "
            f"{sensor['num_taxels']} taxels   |   net |F| "
            f"{sensor.get('total_force', 0.0):.4f} N   |   step {readout.get('steps', 0)}"
            + ("   |   SATURATED" if stats["saturated"] else "")
            + (f"   |   +{others} more sensor(s) not shown" if others else "")
        )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _selected_model(self) -> str:
        if self._model_combo is None:
            return self._models[0]
        index = self._model_combo.model.get_item_value_model().get_value_as_int()
        return self._models[max(0, min(index, len(self._models) - 1))]

    def _scene_force_scale(self):
        """Fixed heatmap scale of the scene on the stage, or None to auto-range.

        Read from the stage rather than the dropdown: what is loaded is what is
        being played, and the two differ the moment someone changes the
        selection without pressing Load Scenario.
        """
        try:
            import omni.usd

            from .scenario import active_scene
        except ImportError:
            return None
        stage = omni.usd.get_context().get_stage()
        scene = active_scene(stage) if stage is not None else None
        return SCENES[scene].force_scale_n if scene is not None else None

    def _selected_scene(self) -> str:
        if self._scene_combo is None:
            return self._scene_keys[0]
        index = self._scene_combo.model.get_item_value_model().get_value_as_int()
        return self._scene_keys[max(0, min(index, len(self._scene_keys) - 1))]

    def _scene_changed(self, *_args) -> None:
        """Show the selected scene's description without building anything."""
        if self._scene_hint_label is not None:
            self._scene_hint_label.text = SCENES[self._selected_scene()].description

    def _spawn_clicked(self) -> None:
        self._set_status(self._on_spawn(self._selected_model()))

    def _load_scenario_clicked(self) -> None:
        self._set_status(self._on_load_scenario(self._selected_model(), self._selected_scene()))

    def _play_one_clicked(self) -> None:
        self._set_status(self._on_play_one())

    def _step_clicked(self, physics_steps: int = 1) -> None:
        self._set_status(self._on_step(physics_steps))

    def _reset_scenario_clicked(self) -> None:
        self._set_status(self._on_reset_scenario())

    def _warm_up_clicked(self) -> None:
        """Kick off the kernel compile on the *next* tick, never on this one.

        **Never call ``app.update()`` from a UI callback.** A button handler runs
        inside an app update already, so re-entering the update loop begins a
        second frame while the first is still recording — the renderer reports
        "make sure you call cmdEnd before calling cmdBegin again", then
        ``acquireNextFrameBufferNoWait`` fails and Kit segfaults. Deferring to a
        one-shot update subscription gets the status message painted without
        re-entering anything.
        """
        if self._warm_up_sub is not None:
            return
        self._set_status("Compiling physics kernels — the app will freeze briefly...")
        self._warm_up_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._run_warm_up, name="synaptics.tactile.warmup")
        )

    def _run_warm_up(self, event) -> None:
        self._warm_up_sub = None  # one-shot
        if self._on_warm_up is not None:
            self._set_status(self._on_warm_up())

    def _diagnostics_clicked(self, verbose: bool) -> None:
        self._set_report(self._on_diagnostics(verbose))

    def _set_status(self, text: str) -> None:
        self._status_text = text or ""
        if self._status_label is not None:
            self._status_label.text = self._status_text

    def _set_headline(self, text: str) -> None:
        self._headline_text = text or ""
        if self._headline_label is not None:
            self._headline_label.text = self._headline_text

    def _set_report(self, text: str) -> None:
        self._report_text = text or ""
        if self._report_label is not None:
            self._report_label.text = self._report_text
