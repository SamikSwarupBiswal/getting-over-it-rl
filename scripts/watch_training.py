"""Show the learner's shared policy in the real game, from the beginning.

Inference is NumPy on CPU. Only this run's trusted, locally produced latest.pkl
is read; no model, archive, rollout or learner STOP file is written. Hidden dots
are actual sampled neurons. Input dots/first-layer edges are signed averages of
groups of normalized network inputs/weights, not six individual input neurons:
pot (including route progress), hammer, cursor/action history, contact/end flags,
smoothed mouse motors, and route/ray sight (including stall/record-gap inputs).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# Limit this viewer's CPU/BLAS usage; it never allocates a second CUDA policy.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goi_jax.training.config import TrainConfig

INPUT_GROUPS = (
    tuple(range(6)) + (15,), tuple(range(6, 11)),
    tuple(range(16, 20)) + tuple(range(95, 103)), tuple(range(11, 15)),
    tuple(range(20, 23)), tuple(range(23, 95)) + (103, 104),
)
CONTRACT_FIELDS = (
    "k", "mouse", "episode_steps", "stall_seconds", "snake_ends", "held_records",
    "time_penalty", "action_cost", "recovery", "regress", "air_bonus", "speed_bonus",
    "norm_obs", "clip_obs", "bounded_mean", "hidden", "section_input", "timed_target", "space_wrap",
)
LOAD_ERRORS = (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, KeyError)


class NumpyPolicy:
    """The same checkpoint actor/normalizer, evaluated without a JAX device call."""

    def __init__(self, checkpoint: dict):
        self.cfg = TrainConfig.from_dict(checkpoint["config"])
        if self.cfg.k != 4:
            raise ValueError("the live preview requires the training k=4 contract")
        if self.cfg.clip_obs <= 0:
            raise ValueError("invalid observation clipping")
        params = checkpoint["params"]
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in params["pi"]]
        self.mu = tuple(np.asarray(x, np.float32) for x in params["mu"])
        self.log_std = np.asarray(params["log_std"], np.float32)
        self.mean, self.var = (np.asarray(x, np.float32) for x in checkpoint["obs_rms"][:2])
        widths = tuple(self.cfg.hidden)
        if len(widths) != 2 or min(widths) < 16 or len(self.layers) != 2:
            raise ValueError("preview HUD requires two actor layers with at least 16 neurons")
        self.obs_size = 105 + int(self.cfg.timed_target > 0)
        self.groups = INPUT_GROUPS + ((105,),) if self.obs_size == 106 else INPUT_GROUPS
        expected = ((self.obs_size, widths[0]), (widths[0], widths[1]))
        for (w, b), shape in zip(self.layers, expected):
            if w.shape != shape or b.shape != (shape[1],):
                raise ValueError("actor layer shape does not match its observation/config contract")
        if self.mu[0].shape != (widths[1], 2) or self.mu[1].shape != (2,) or self.log_std.shape != (2,):
            raise ValueError("preview requires exactly two mouse outputs")
        if self.mean.shape != (self.obs_size,) or self.var.shape != (self.obs_size,) or np.any(self.var < 0):
            raise ValueError(f"invalid {self.obs_size}-input observation normalizer")
        arrays = [self.mean, self.var, self.log_std, *self.mu]
        arrays += [a for pair in self.layers for a in pair]
        self.section_bias = None
        if "pi_sec" in params:
            from goi_jax.training.starts import SECTION_LO
            self.section_lows = np.asarray(SECTION_LO)
            self.section_bias = np.asarray(params["pi_sec"], np.float32)
            if self.section_bias.shape != (len(self.section_lows), widths[0]):
                raise ValueError("invalid section-input actor")
            arrays.append(self.section_bias)
        if bool(self.section_bias is not None) != bool(self.cfg.section_input):
            raise ValueError("section-input config and actor disagree")
        if not all(np.isfinite(a).all() for a in arrays) or not np.isfinite(np.exp(self.log_std)).all():
            raise ValueError("non-finite actor/normalizer")
        self.signature = tuple(getattr(self.cfg, name) for name in CONTRACT_FIELDS)
        self.indices = [np.linspace(0, width - 1, 16, dtype=int) for width in widths]
        i, j = self.indices
        self.edges = dict(
            edges1=np.stack([self.layers[0][0][list(group)][:, i].mean(axis=0)
                             for group in self.groups]).ravel().tolist(),
            edges2=self.layers[1][0][np.ix_(i, j)].ravel().tolist(),
            edges3=self.mu[0][j].ravel().tolist(),
        )
        self.update = int(checkpoint.get("update", 0))
        self.steps = int(checkpoint.get("env_steps", 0))
        self.wall_s = float(checkpoint.get("wall_s", 0.0))

    def forward(self, raw: np.ndarray) -> tuple[np.ndarray, dict]:
        raw = np.asarray(raw, np.float32)
        if raw.shape != (self.obs_size,) or not np.isfinite(raw).all():
            raise ValueError(f"game did not produce {self.obs_size} finite policy inputs")
        x = (np.clip((raw - self.mean) / np.sqrt(self.var + np.float32(1e-8)),
                     -self.cfg.clip_obs, self.cfg.clip_obs) if self.cfg.norm_obs else raw)
        inputs = [float(x[list(group)].mean()) for group in self.groups]
        hidden = []
        for index, (w, b) in enumerate(self.layers):
            z = x @ w + b
            if index == 0 and self.section_bias is not None:
                sec = int(np.clip(np.searchsorted(self.section_lows, raw[15], side="right") - 1,
                                  0, len(self.section_lows) - 1))
                z = z + self.section_bias[sec]
            x = np.tanh(z)
            hidden.append(x[self.indices[index]].tolist())
        mean = x @ self.mu[0] + self.mu[1]
        if self.cfg.bounded_mean:
            mean = np.tanh(mean)
        return mean, dict(inputValues=inputs, hidden1=hidden[0], hidden2=hidden[1], **self.edges)

    def act(self, raw: np.ndarray, rng: np.random.Generator, noise: float = 0.5) -> tuple[np.ndarray, dict]:
        mean, visual = self.forward(raw)
        action = np.clip(mean + noise * np.exp(self.log_std) * rng.standard_normal(2), -1, 1).astype(np.float32)
        visual["outputs"] = action.tolist()
        return action, visual


class CheckpointReader:
    """Adopt an entire atomic save, retaining the previous actor on an incompatible/failed reload."""

    def __init__(self, path: Path):
        self.path, self.policy, self.stamp, self.mtime = Path(path), None, None, 0.0

    def refresh(self) -> bool:
        before = self.path.stat()
        stamp = (before.st_mtime_ns, before.st_size, before.st_ino)
        if stamp == self.stamp:
            return False
        # This is the user's own local trainer checkpoint, not an arbitrary downloaded pickle.
        with self.path.open("rb") as stream:
            candidate = NumpyPolicy(pickle.load(stream))
        after = self.path.stat()
        if stamp != (after.st_mtime_ns, after.st_size, after.st_ino):
            return False  # another atomic replacement arrived during the read; retry next poll
        if self.policy is not None and candidate.signature != self.policy.signature:
            raise ValueError("new checkpoint changes the active observation/action/episode contract")
        self.policy, self.stamp, self.mtime = candidate, stamp, before.st_mtime
        return True


def latest_metrics(run: Path) -> dict:
    """Read the newest complete JSONL row, including the brief gap during log rotation."""
    for name in ("metrics.jsonl", "metrics.1.jsonl", "metrics.2.jsonl"):
        try:
            with (run / name).open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - 65536))
                tail = stream.read()
            # The trainer terminates each complete row with a newline; ignore any partial final write.
            for line in reversed(tail.split(b"\n")[:-1]):
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "env_steps" in row:
                        return row
                except (ValueError, UnicodeDecodeError):
                    continue
        except FileNotFoundError:
            continue
    return {}


def training_metrics(reader: CheckpointReader, row: dict) -> dict:
    policy = reader.policy
    recent_save = 0 <= time.time() - reader.mtime <= 30.0
    def number(key, fallback):
        value = float(row.get(key, fallback))
        return value if math.isfinite(value) else fallback
    return dict(update=int(number("update", policy.update)), policyUpdate=policy.update,
                trainingActions=int(number("env_steps", policy.steps)),
                actionsPerSecond=number("env_steps_per_s", 0.0) if recent_save else 0.0,
                trainingStatus="recent saves" if recent_save else "no recent save",
                elapsed=number("wall_s", policy.wall_s), parallelGames=int(policy.cfg.num_envs))


class RouteHistory:
    """A full attempt on a game-time axis, thinning older samples as it grows."""

    def __init__(self):
        self.samples = []
        self.episode = None

    def sample(self, episode: int, now: float, progress: float) -> dict:
        if episode != self.episode:
            self.samples.clear()
            self.episode, self.spacing = episode, 0.5
        elapsed = now
        if math.isfinite(progress) and (not self.samples or elapsed - self.samples[-1][0] >= self.spacing):
            self.samples.append((elapsed, float(progress)))
            if len(self.samples) >= 257:
                self.samples = self.samples[::2]
                self.spacing *= 2
        points = self.samples
        if points and elapsed > points[-1][0] and math.isfinite(progress):
            points = points + [(elapsed, float(progress))]
        return dict(previewSeconds=elapsed, progressTimes=[t for t, _ in points],
                    progressHistory=[p for _, p in points])


def update_records(records, episode, peak, target_seconds):
    if peak > records.get("recordDistance", -1):
        records.update(recordDistance=float(peak), distanceAttempt=episode)
    fastest = records.get("fastestSeconds", 0)
    if target_seconds is not None and (fastest <= 0 or target_seconds < fastest):
        records.update(fastestSeconds=target_seconds, fastestAttempt=episode)


def preview_end(reason, idle_seconds):
    if reason in ("lake", "won", "snake"):
        return reason
    return "30s without new headway" if idle_seconds >= 30 else None


def saved_attempt(path: Path) -> int:
    """Resume the last published attempt across viewer restarts."""
    try:
        frame = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    attempt = frame.get("episode") if isinstance(frame, dict) else None
    return attempt if type(attempt) is int and 1 <= attempt <= 2**31 - 1 else 1


def write_hud(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".preview-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
        # Windows readers can briefly deny replacement of files on the WSL mount.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.02 * 2**attempt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/fresh-gpu")
    parser.add_argument("--noise", type=float, default=0.5, help="Gaussian sampling multiplier; 0 uses the actor mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reload-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.noise) or args.noise < 0 or not math.isfinite(args.reload_seconds) or args.reload_seconds < 0.1:
        parser.error("noise must be nonnegative and reload-seconds must be at least 0.1 (both finite)")
    return args


@contextmanager
def preview_lock(path: Path):
    """One visible preview per project; closing the owner releases the WSL flock."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv=None) -> int:
    args = parse(argv)
    with preview_lock(ROOT / "runtime/live-preview.lock") as acquired:
        if not acquired:
            print("The live preview is already running.", flush=True)
            return 0
        return run_preview(args)


def run_preview(args) -> int:
    hud, stop = ROOT / "runtime/live-preview.json", ROOT / "runtime/PREVIEW_STOP"
    episode = saved_attempt(hud)
    stopped = False
    def request_stop(_sig, _frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    reader = CheckpointReader(args.run.resolve() / "checkpoints/latest.pkl")
    print(f"LIVE PREVIEW: shared policy {reader.path}; stop with {stop}", flush=True)
    while not stopped and not stop.exists():
        try:
            reader.refresh()
            if reader.policy is not None:
                break
        except LOAD_ERRORS as exc:
            print(f"Waiting for the learner's checkpoint: {exc}", flush=True)
        time.sleep(1)
    if stopped or stop.exists():
        return 0
    # Lazy CPU-only imports keep --help and NumPy inference tests free of JAX initialization.
    from goi_jax import bridge
    from goi_jax.play import HostEpisode
    win_hud = subprocess.check_output(["wslpath", "-w", str(hud.resolve())], text=True).strip()
    rng = np.random.default_rng(args.seed)
    proc = client = None
    payload, error = {}, ""
    def launch_game():
        flags = ("-goi-hud", win_hud, "-goi-audio", "-goi-ending")
        if not reader.policy.cfg.space_wrap:
            flags += ("-goi-no-space-wrap",)
        return bridge.start(headless=False, width=1920, height=1080, fullscreen=True,
                            args=flags, k=reader.policy.cfg.k,
                            n_rays=32, ray_length=15.0, head_rays=0, head_length=4.0, render=True)

    def close_game():
        nonlocal proc, client
        try:
            if client is not None:
                client.sock.settimeout(3)
                client.close()
        finally:
            client = None
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=5)
                proc = None

    try:
        cfg = reader.policy.cfg
        proc, client = launch_game()
        ep = HostEpisode(cfg, client.reset())
        records_path = args.run / f"preview-records-{cfg.timed_target:g}.json"
        records = json.loads(records_path.read_text()) if records_path.exists() else {}
        persisted = records.copy()
        next_save = 0.0
        history = RouteHistory()
        history.sample(episode, 0.0, ep.progress)
        mark, mark_step = ep.progress, 0
        end = None
        tick, deadline, next_reload, next_hud = cfg.k / 120.0, time.monotonic(), 0.0, 0.0
        while not stopped and not stop.exists():
            now = time.monotonic()
            if now >= next_reload:
                try:
                    if reader.refresh():
                        print(f"Shared policy refreshed: update {reader.policy.update:,}", flush=True)
                    error = ""
                except LOAD_ERRORS as exc:
                    error = f"Checkpoint reload: {exc}"
                    print(error, flush=True)
                next_reload = now + args.reload_seconds
            if end:
                ep = HostEpisode(reader.policy.cfg, client.reset())
                episode += 1
                history.sample(episode, 0.0, ep.progress)
                mark, mark_step = ep.progress, 0
                end = None
                next_hud = 0.0
            observation = ep.obs()
            observation[103] = min(1.0, observation[103])
            action, visual = reader.policy.act(observation, rng, args.noise)
            raw = client.step(*(action * cfg.mouse).astype(float))
            target_seconds = ep.target_seconds
            _, end = ep.step(action, raw, client.on_snake)
            if ep.progress >= mark + 1.0:
                mark, mark_step = ep.progress, ep.t
            end = preview_end(end, (ep.t - mark_step) * tick)
            update_records(records, episode, ep.best, ep.target_seconds)
            if end:
                records["lastReset"] = f"#{episode} {end} at {ep.progress:.1f} ({ep.t * tick:.1f}s)"
                print(f"Preview reset: {records['lastReset']}", flush=True)
            if target_seconds is None and ep.target_seconds is not None:
                print(f"Attempt {episode}: reached route {cfg.timed_target:g} in {ep.target_seconds:.2f} game seconds",
                      flush=True)
            now = time.monotonic()
            if records != persisted and (now >= next_save or end or target_seconds != ep.target_seconds):
                write_hud(records_path, records)
                persisted = records.copy()
                next_save = now + 1.0
            if now >= next_hud or end:
                payload = dict(phase="LIVE PREVIEW / shared policy", error=error,
                               **training_metrics(reader, latest_metrics(args.run)),
                               previewProgress=float(ep.progress), previewBest=float(ep.best),
                               previewReturn=float(ep.ret), episode=episode,
                               targetProgress=cfg.timed_target, targetSeconds=ep.target_seconds or 0.0,
                               modelAge=max(0.0, time.time() - reader.mtime),
                               **history.sample(episode, ep.t * tick, float(ep.progress)), **records, **visual)
                write_hud(hud, payload)
                next_hud = now + 0.1
            deadline = max(deadline + tick, time.monotonic())
            time.sleep(max(0.0, deadline - time.monotonic()))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        # Disconnecting sends the plugin's quit command. Never stop another process by executable name.
        try:
            close_game()
        finally:
            if payload:
                if records != persisted:
                    write_hud(records_path, records)
                write_hud(hud, dict(payload, phase="PREVIEW STOPPED", error=error))


if __name__ == "__main__":
    raise SystemExit(main())
