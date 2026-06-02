import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# Try to force an interactive backend on Windows/Anaconda.
# If Tk is unavailable, Matplotlib will fall back to its configured backend.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.widgets import Button


# Keep important GUI objects alive for the whole program.
_VISUALIZER = None


DEFAULT_THRUST_COLUMN_NAMES = [
    "thrust",
    "T",
    "u",
    "force",
    "F",
    "thrust_force",
    "thrust_cmd",
    "motor_thrust",
]


def find_column(df, names, required=True, default=None):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for name in names:
        key = str(name).strip().lower()
        if key in lookup:
            return df[lookup[key]].to_numpy(dtype=float), lookup[key]

    if required:
        raise ValueError(
            f"Missing required column. Tried: {names}\n"
            f"Available columns: {list(df.columns)}"
        )

    return default, None


def body_axes(theta, theta_sign=1.0):
    """
    Body frame convention:

        theta = 0:
            body x-axis points right
            body y-axis points up
            thrust/exhaust direction points down

    Positive theta rotates the body counterclockwise when theta_sign = 1.
    """
    th = theta_sign * theta

    x_axis = np.array([np.cos(th), np.sin(th)])
    y_axis = np.array([-np.sin(th), np.cos(th)])
    down_axis = -y_axis

    return x_axis, y_axis, down_axis


def thrust_to_arrow_length(thrust_value, thrust_min, thrust_max, visual_min, visual_max):
    """
    Maps thrust force to a visible arrow length.

    By default, thrust is assumed to be in the range 0 to 20.
    Values outside that range are clipped for visualization only.
    """
    if thrust_max <= thrust_min:
        raise ValueError("--thrust-max must be greater than --thrust-min.")

    thrust_clipped = np.clip(thrust_value, thrust_min, thrust_max)
    normalized = (thrust_clipped - thrust_min) / (thrust_max - thrust_min)
    return visual_min + normalized * (visual_max - visual_min)


def geometry(
    x,
    y,
    theta,
    d,
    thrust_value,
    body_length,
    body_width,
    theta_sign,
    d_sign,
    position_is_com,
    thrust_min,
    thrust_max,
    thrust_visual_min,
    thrust_visual_max,
):
    """
    d is interpreted as a COM offset along the drone's local/body x-axis.
    The thrust vector is drawn along body-down, with length scaled by thrust_value.
    """
    x_axis, y_axis, down_axis = body_axes(theta, theta_sign)

    p = np.array([x, y], dtype=float)
    com_offset = d_sign * d * x_axis

    if position_is_com:
        com = p
        ref = com - com_offset
    else:
        ref = p
        com = ref + com_offset

    half_l = body_length / 2.0
    half_w = body_width / 2.0

    corners = np.array(
        [
            ref + half_l * x_axis + half_w * y_axis,
            ref - half_l * x_axis + half_w * y_axis,
            ref - half_l * x_axis - half_w * y_axis,
            ref + half_l * x_axis - half_w * y_axis,
        ]
    )

    thrust_length = thrust_to_arrow_length(
        thrust_value,
        thrust_min,
        thrust_max,
        thrust_visual_min,
        thrust_visual_max,
    )

    thrust_start = ref
    thrust_end = ref + thrust_length * down_axis

    nose_start = ref
    nose_end = ref + 0.45 * body_length * x_axis

    return ref, com, corners, thrust_start, thrust_end, nose_start, nose_end, thrust_length


def maximize_figure_window(fig):
    """
    Best-effort maximize for common Matplotlib GUI backends.
    This avoids full-screen mode and simply tries to start maximized.
    """
    manager = fig.canvas.manager
    maximized = False

    # TkAgg on Windows/Linux
    try:
        manager.window.state("zoomed")
        maximized = True
    except Exception:
        pass

    # Qt backends
    if not maximized:
        try:
            manager.window.showMaximized()
            maximized = True
        except Exception:
            pass

    # wx backend
    if not maximized:
        try:
            manager.frame.Maximize(True)
            maximized = True
        except Exception:
            pass

    # Tk fallback: resize to maximum available size
    if not maximized:
        try:
            max_w, max_h = manager.window.maxsize()
            manager.resize(max_w, max_h)
            maximized = True
        except Exception:
            pass

    return maximized


class DroneVisualizer:
    def __init__(self, args, data):
        self.args = args
        self.t = data["t"]
        self.x = data["x"]
        self.y = data["y"]
        self.theta = data["theta"]
        self.d = data["d"]
        self.thrust = data["thrust"]
        self.thrust_column_found = data["thrust_column_found"]

        self.t0 = float(self.t[0])
        self.tf = float(self.t[-1])
        self.duration = self.tf - self.t0

        if args.real_time:
            self.effective_speed = 1.0
        else:
            self.effective_speed = args.speed

        self.start_wall_time = None
        self.current_sim_time = self.t0
        self.finished = False

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        try:
            self.fig.canvas.manager.set_window_title("Drone CSV 2D Visualizer - Real-Time Capable")
        except Exception:
            pass

        # Leave space for the replay button.
        self.fig.subplots_adjust(bottom=0.15)

        self.ax.set_title("Drone 2D Simulation Playback")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)

        span_x = float(np.max(self.x) - np.min(self.x))
        span_y = float(np.max(self.y) - np.min(self.y))
        pad = max(
            args.body_length * 2.0,
            args.thrust_visual_max * 1.5,
            0.15 * max(span_x, span_y, 1.0),
        )

        self.fixed_xlim = (float(np.min(self.x) - pad), float(np.max(self.x) + pad))
        self.fixed_ylim = (float(np.min(self.y) - pad), float(np.max(self.y) + pad))
        self.ax.set_xlim(*self.fixed_xlim)
        self.ax.set_ylim(*self.fixed_ylim)

        # Initial artist setup.
        x0, y0, theta0, d0, thrust0 = self.sample_at(self.t0)
        ref, com, corners, thrust_start, thrust_end, nose_start, nose_end, _ = geometry(
            x0,
            y0,
            theta0,
            d0,
            thrust0,
            args.body_length,
            args.body_width,
            args.theta_sign,
            args.d_sign,
            args.position_is_com,
            args.thrust_min,
            args.thrust_max,
            args.thrust_visual_min,
            args.thrust_visual_max,
        )

        self.body_patch = Polygon(corners, closed=True, fill=False, linewidth=2)
        self.ax.add_patch(self.body_patch)

        self.com_patch = Circle(com, args.com_radius, fill=True)
        self.ax.add_patch(self.com_patch)

        self.ref_marker, = self.ax.plot(
            [ref[0]],
            [ref[1]],
            marker="o",
            markersize=4,
            linestyle="None",
            label="reference",
        )
        self.ax.plot([], [], marker="o", linestyle="None", label="COM")
        self.trajectory_line, = self.ax.plot([], [], linewidth=1.5, label="trajectory")
        self.nose_line, = self.ax.plot(
            [nose_start[0], nose_end[0]],
            [nose_start[1], nose_end[1]],
            linewidth=2,
            label="body x-axis",
        )

        self.thrust_arrow = FancyArrowPatch(
            thrust_start,
            thrust_end,
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2,
            label="thrust vector",
        )
        self.ax.add_patch(self.thrust_arrow)

        self.hud = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        self.ax.legend(loc="upper right")

        # Replay button in normalized figure coordinates: [left, bottom, width, height]
        self.replay_ax = self.fig.add_axes([0.42, 0.035, 0.16, 0.055])
        self.replay_button = Button(self.replay_ax, "Replay")
        self.replay_button.on_clicked(self.replay)

        # Timer-based animation. This is more reliable for real-time playback than
        # stepping through every frame with FuncAnimation.
        interval_ms = max(1, int(round(1000.0 / args.max_fps)))
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.on_timer)

        self.update_plot(self.t0)

    def sample_at(self, sim_time):
        sim_time = float(np.clip(sim_time, self.t0, self.tf))
        x_val = float(np.interp(sim_time, self.t, self.x))
        y_val = float(np.interp(sim_time, self.t, self.y))
        theta_val = float(np.interp(sim_time, self.t, self.theta))
        d_val = float(np.interp(sim_time, self.t, self.d))
        thrust_val = float(np.interp(sim_time, self.t, self.thrust))
        return x_val, y_val, theta_val, d_val, thrust_val

    def update_plot(self, sim_time):
        self.current_sim_time = float(np.clip(sim_time, self.t0, self.tf))
        args = self.args

        x_val, y_val, theta_val, d_val, thrust_val = self.sample_at(self.current_sim_time)

        ref, com, corners, thrust_start, thrust_end, nose_start, nose_end, thrust_length = geometry(
            x_val,
            y_val,
            theta_val,
            d_val,
            thrust_val,
            args.body_length,
            args.body_width,
            args.theta_sign,
            args.d_sign,
            args.position_is_com,
            args.thrust_min,
            args.thrust_max,
            args.thrust_visual_min,
            args.thrust_visual_max,
        )

        self.body_patch.set_xy(corners)
        self.com_patch.center = com
        self.ref_marker.set_data([ref[0]], [ref[1]])
        self.nose_line.set_data([nose_start[0], nose_end[0]], [nose_start[1], nose_end[1]])
        self.thrust_arrow.set_positions(thrust_start, thrust_end)

        if args.trail > 0:
            j0 = np.searchsorted(self.t, self.current_sim_time - args.trail, side="left")
        else:
            j0 = 0

        j1 = np.searchsorted(self.t, self.current_sim_time, side="right")
        trace_x = np.append(self.x[j0:j1], x_val)
        trace_y = np.append(self.y[j0:j1], y_val)
        self.trajectory_line.set_data(trace_x, trace_y)

        if args.follow:
            window = max(args.body_length * 4.0, args.thrust_visual_max * 3.0, 1.0)
            self.ax.set_xlim(ref[0] - window, ref[0] + window)
            self.ax.set_ylim(ref[1] - window, ref[1] + window)
        else:
            self.ax.set_xlim(*self.fixed_xlim)
            self.ax.set_ylim(*self.fixed_ylim)

        thrust_column_text = (
            self.thrust_column_found if self.thrust_column_found is not None else "constant fallback"
        )
        mode_text = "real-time" if args.real_time else f"{self.effective_speed:.2f}x"

        self.hud.set_text(
            f"sim t = {self.current_sim_time:.3f} s\n"
            f"x = {x_val:.3f}\n"
            f"y = {y_val:.3f}\n"
            f"theta = {theta_val:.3f} rad\n"
            f"d = {d_val:.3f}\n"
            f"thrust = {thrust_val:.3f} / {args.thrust_max:g}\n"
            f"arrow length = {thrust_length:.3f}\n"
            f"thrust col = {thrust_column_text}\n"
            f"playback = {mode_text}"
        )

        self.fig.canvas.draw_idle()

    def start(self):
        self.start_wall_time = time.perf_counter()
        self.finished = False
        self.timer.start()

    def replay(self, _event=None):
        self.start_wall_time = time.perf_counter()
        self.current_sim_time = self.t0
        self.finished = False
        self.update_plot(self.t0)
        self.timer.start()

    def on_timer(self):
        if self.finished:
            return False

        if self.start_wall_time is None:
            self.start_wall_time = time.perf_counter()

        elapsed_wall = time.perf_counter() - self.start_wall_time
        target_sim_time = self.t0 + elapsed_wall * self.effective_speed

        if target_sim_time >= self.tf:
            self.update_plot(self.tf)
            self.finished = True
            self.timer.stop()
            return False

        self.update_plot(target_sim_time)
        return True

    def show(self):
        if not self.args.no_maximize:
            maximize_figure_window(self.fig)
        self.start()
        plt.show(block=True)


def load_data(args):
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    t, _ = find_column(df, ["time", "t"])
    x, _ = find_column(df, ["x"])
    y, _ = find_column(df, ["y"])
    theta, _ = find_column(df, ["theta", "angle", "pitch"])
    d, _ = find_column(df, ["d", "com_offset", "offset"], required=False)

    if d is None:
        d = np.zeros_like(t)

    if args.thrust_column:
        thrust, thrust_column_found = find_column(df, [args.thrust_column], required=True)
    else:
        thrust, thrust_column_found = find_column(
            df,
            DEFAULT_THRUST_COLUMN_NAMES,
            required=False,
        )

    if thrust is None:
        # The user's current CSV may only have time, x, y, theta, d.
        # Use constant max thrust for visualization so the script still runs.
        thrust = np.full_like(t, args.thrust_max, dtype=float)
        thrust_column_found = None
        print(
            "WARNING: No thrust column found. Drawing constant max thrust.\n"
            f"Expected one of {DEFAULT_THRUST_COLUMN_NAMES}, or pass --thrust-column YourColumnName.\n"
            "To animate changing thrust length, add thrust to your CSV export.",
            file=sys.stderr,
        )

    if args.degrees:
        theta = np.deg2rad(theta)

    finite_mask = (
        np.isfinite(t)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(theta)
        & np.isfinite(d)
        & np.isfinite(thrust)
    )
    t = t[finite_mask]
    x = x[finite_mask]
    y = y[finite_mask]
    theta = theta[finite_mask]
    d = d[finite_mask]
    thrust = thrust[finite_mask]

    if len(t) < 2:
        raise ValueError("CSV must contain at least two valid time samples.")

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]
    theta = theta[order]
    d = d[order]
    thrust = thrust[order]

    # Interpolation needs unique time values. Keep the first sample for duplicates.
    t_unique, unique_indices = np.unique(t, return_index=True)
    t = t_unique
    x = x[unique_indices]
    y = y[unique_indices]
    theta = theta[unique_indices]
    d = d[unique_indices]
    thrust = thrust[unique_indices]

    if len(t) < 2:
        raise ValueError("CSV must contain at least two unique time samples.")

    if not np.all(np.diff(t) > 0):
        raise ValueError("Time column must be strictly increasing after cleanup.")

    return {
        "t": t,
        "x": x,
        "y": y,
        "theta": theta,
        "d": d,
        "thrust": thrust,
        "thrust_column_found": thrust_column_found,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time 2D playback for a Simulink drone CSV log."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="drone_log.csv",
        help="CSV file exported from MATLAB/Simulink. Default: drone_log.csv",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help=(
            "Wall-clock synchronized playback: 1 second of simulation time takes "
            "1 second in the real world. This forces effective speed to 1.0."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Playback speed multiplier when --real-time is not used. "
            "Example: --speed 2 plays twice as fast. Default: 1"
        ),
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=60.0,
        help="Maximum display frame rate. Default: 60",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Use this if theta is stored in degrees instead of radians.",
    )
    parser.add_argument(
        "--theta-sign",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Use -1 if the drone appears to rotate backwards.",
    )
    parser.add_argument(
        "--d-sign",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Use -1 if the COM offset appears on the wrong side.",
    )
    parser.add_argument(
        "--position-is-com",
        action="store_true",
        help="Use this if x,y in the CSV are already the COM position.",
    )
    parser.add_argument(
        "--body-length",
        type=float,
        default=0.60,
        help="Visual body length in plot units. Default: 0.60",
    )
    parser.add_argument(
        "--body-width",
        type=float,
        default=0.18,
        help="Visual body width in plot units. Default: 0.18",
    )
    parser.add_argument(
        "--com-radius",
        type=float,
        default=0.045,
        help="Visual COM marker radius. Default: 0.045",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep camera centered on the drone instead of using fixed axes.",
    )
    parser.add_argument(
        "--trail",
        type=float,
        default=0.0,
        help="Trail duration in seconds. Use 0 for full trajectory. Default: 0",
    )
    parser.add_argument(
        "--thrust-column",
        default=None,
        help=(
            "Name of the thrust column in the CSV. If omitted, the script tries "
            f"these names: {DEFAULT_THRUST_COLUMN_NAMES}"
        ),
    )
    parser.add_argument(
        "--thrust-min",
        type=float,
        default=0.0,
        help="Minimum thrust value for scaling. Default: 0",
    )
    parser.add_argument(
        "--thrust-max",
        type=float,
        default=20.0,
        help="Maximum thrust value for scaling. Default: 20",
    )
    parser.add_argument(
        "--thrust-visual-min",
        type=float,
        default=0.05,
        help="Arrow length when thrust is at minimum. Default: 0.05",
    )
    parser.add_argument(
        "--thrust-visual-max",
        type=float,
        default=0.85,
        help="Arrow length when thrust is at maximum. Default: 0.85",
    )
    parser.add_argument(
        "--no-maximize",
        action="store_true",
        help="Disable maximize-on-start.",
    )
    return parser


def validate_args(args):
    if args.speed <= 0:
        raise ValueError("--speed must be greater than zero.")
    if args.max_fps <= 0:
        raise ValueError("--max-fps must be greater than zero.")
    if args.trail < 0:
        raise ValueError("--trail cannot be negative.")
    if args.body_length <= 0:
        raise ValueError("--body-length must be greater than zero.")
    if args.body_width <= 0:
        raise ValueError("--body-width must be greater than zero.")
    if args.com_radius <= 0:
        raise ValueError("--com-radius must be greater than zero.")
    if args.thrust_max <= args.thrust_min:
        raise ValueError("--thrust-max must be greater than --thrust-min.")
    if args.thrust_visual_max <= args.thrust_visual_min:
        raise ValueError("--thrust-visual-max must be greater than --thrust-visual-min.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    data = load_data(args)

    global _VISUALIZER
    _VISUALIZER = DroneVisualizer(args, data)
    _VISUALIZER.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
