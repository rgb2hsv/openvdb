#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Build, run and render every MOGAC unit-test simulation.

Steps (each can be skipped):
  1. build   cmake --build <build-dir> --target vdb_test
  2. tests   run vdb_test --gtest_filter=<filter> in parallel shards with
             OPENVDB_MOGAC_FRAME_DIR=<out>/frames, so every evolver of every
             test writes one PLY per step (see unittest/LevelSetMultiObjectFrames.h)
  3. render  for every simulation <sim> = "<TestName>-<k>":
               <out>/<sim>/png/frame_%05d.png   one image per step, fixed camera
               <out>/<sim>/<sim>.mp4            the frames as an H.264 video
               <out>/<sim>/meshes/initial.ply   first frame's mesh
               <out>/<sim>/meshes/midway.ply    middle frame's mesh
               <out>/<sim>/meshes/final.ply     last frame's mesh
             then deletes <out>/frames unless --keep-frames.

Each object (label) has its own color, fixed for the whole simulation. The
meshes come from tools::multiObjectToMesh with the evolver's digital
connectivity, so topology-preserving runs are meshed consistently with it.

Requires numpy and pyvista for rendering and ffmpeg for the videos. If pyvista
is missing, the script re-runs itself with <out>/.venv/bin/python (or the one\nunder the default output root) when present.
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
DEFAULT_OUT = "/mnt/storage/scratch/openvdb"
DEFAULT_FILTER = "TestLevelSetMultiObject*"
FRAME_ENV = "OPENVDB_MOGAC_FRAME_DIR"
IMAGE_SIZE = (1280, 960)
BACKGROUND = (0.88, 0.88, 0.9)
BOX_COLOR = (0.6, 0.6, 0.65)
VIEW_DIRECTION = (1.0, -1.3, 0.9)   # camera direction from the scene center
CAMERA_DISTANCE = 3.3               # in bounding-sphere radii
VIEW_ANGLE_DEG = 30.0
NEAR_CLIP, FAR_CLIP = 0.1, 10.0     # in bounding-sphere radii
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 8
KEY_FRAMES = ("initial", "midway", "final")


# ----------------------------------------------------------------------------
# Steps 1 and 2: build and run the tests


def build(build_dir, jobs):
    print(f"[build] {build_dir}", flush=True)
    subprocess.run(["cmake", "--build", build_dir, "--target", "vdb_test", "-j", str(jobs)],
                   check=True)


def run_tests(build_dir, gtest_filter, shards, frame_root, log_dir):
    exe = os.path.join(build_dir, "openvdb", "openvdb", "unittest", "vdb_test")
    if not os.path.exists(exe):
        sys.exit(f"vdb_test not found at {exe} (build first or pass --build-dir)")
    shutil.rmtree(frame_root, ignore_errors=True)
    os.makedirs(frame_root)
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, **{FRAME_ENV: frame_root, "GTEST_TOTAL_SHARDS": str(shards)})
    print(f"[tests] {gtest_filter} in {shards} shard(s), frames to {frame_root}", flush=True)
    start = time.time()
    procs = []
    for k in range(shards):
        log = open(os.path.join(log_dir, f"shard_{k}.txt"), "w")
        procs.append((subprocess.Popen([exe, f"--gtest_filter={gtest_filter}"],
                                       env=dict(env, GTEST_SHARD_INDEX=str(k)),
                                       stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for k, (p, log) in enumerate(procs):
        p.wait()
        log.close()
        with open(log.name) as f:
            for line in f:
                if line.startswith("[  FAILED  ]") and "," not in line and "listed below" not in line:
                    failed.append(line.split()[3])
    print(f"[tests] done in {time.time() - start:.0f} s; "
          f"{len(set(failed))} failed{': ' + ', '.join(sorted(set(failed))) if failed else ''}",
          flush=True)
    return sorted(set(failed))


# ----------------------------------------------------------------------------
# Step 3: render


def load_ply(path):
    """Read a frame PLY: binary little endian, float xyz, uchar rgb, int label,
    triangles. Returns (points, faces, rgb, label)."""
    import numpy as np
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    nv = nf = 0
    for line in data[:end].decode().split("\n"):
        if line.startswith("element vertex"):
            nv = int(line.split()[2])
        elif line.startswith("element face"):
            nf = int(line.split()[2])
    vdt = np.dtype([("p", "<f4", 3), ("c", "u1", 3), ("l", "<i4")])
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    verts = np.frombuffer(data, vdt, nv, end)
    faces = np.frombuffer(data, fdt, nf, end + nv * vdt.itemsize)
    return verts["p"].astype(np.float64), faces["i"], verts["c"], verts["l"]


def read_times(sim_dir):
    times = {}
    path = os.path.join(sim_dir, "frames.txt")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    times[int(parts[0])] = float(parts[1])
    return times


class VideoWriter:
    """Streams raw RGB frames to ffmpeg. Output goes through a pipe because a
    snap-packaged ffmpeg cannot open files outside $HOME."""

    def __init__(self, path, size, fps):
        ffmpeg = shutil.which("ffmpeg") or "/snap/bin/ffmpeg"
        self.out = open(path, "wb")
        self.proc = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.reader = threading.Thread(target=shutil.copyfileobj, args=(self.proc.stdout, self.out))
        self.reader.start()

    def write(self, rgb):
        self.proc.stdin.write(rgb.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.reader.join()
        self.out.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def render_sim(sim_dir, out_dir, fps):
    import numpy as np
    os.environ.setdefault("DISPLAY", ":1")
    import pyvista as pv

    sim = os.path.basename(os.path.normpath(sim_dir))
    files = sorted(f for f in os.listdir(sim_dir) if f.startswith("frame_") and f.endswith(".ply"))
    if not files:
        return sim, 0
    times = read_times(sim_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    png_dir = os.path.join(out_dir, "png")
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(png_dir)
    os.makedirs(mesh_dir)

    # initial / midway / final meshes, copied as written by the test
    for key, idx in zip(KEY_FRAMES, (0, len(files) // 2, len(files) - 1)):
        shutil.copyfile(os.path.join(sim_dir, files[idx]), os.path.join(mesh_dir, key + ".ply"))

    # fixed camera: union of the bounds of all frames; legend of all labels
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    legend = {}
    for name in files:
        p, _, rgb, lab = load_ply(os.path.join(sim_dir, name))
        if len(p):
            lo = np.minimum(lo, p.min(0))
            hi = np.maximum(hi, p.max(0))
            for label in np.unique(lab):
                legend.setdefault(int(label), tuple(int(v) for v in rgb[np.argmax(lab == label)]))
    if not np.all(np.isfinite(lo)):
        lo, hi = -np.ones(3), np.ones(3)
    center = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-6)
    direction = np.asarray(VIEW_DIRECTION) / np.linalg.norm(VIEW_DIRECTION)

    pl = pv.Plotter(off_screen=True, window_size=IMAGE_SIZE)
    pl.set_background(BACKGROUND)
    pl.enable_anti_aliasing("ssaa")
    pl.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])),
                style="wireframe", color=BOX_COLOR, line_width=1)
    entries = [[f"label {l}", tuple(c / 255.0 for c in legend[l])] for l in sorted(legend)]
    if entries:
        pl.add_legend(entries, bcolor=None, border=False, loc="upper right",
                      size=(0.09, 0.022 * len(entries) + 0.01), face="rectangle")
        for actor in pl.renderer.actors.values():
            if actor.__class__.__name__ == "vtkLegendBoxActor":
                actor.GetEntryTextProperty().SetFontSize(LEGEND_FONT_SIZE)
                actor.GetEntryTextProperty().SetFontFamilyToArial()

    video = None
    mesh_actor = text = None
    for n, name in enumerate(files):
        p, f, rgb, _ = load_ply(os.path.join(sim_dir, name))
        if mesh_actor is not None:
            pl.remove_actor(mesh_actor, render=False)
            mesh_actor = None
        if len(p) and len(f):
            cells = np.hstack([np.full((len(f), 1), 3, np.int64), f.astype(np.int64)]).ravel()
            mesh = pv.PolyData(p, cells)
            mesh.point_data["rgb"] = rgb
            mesh_actor = pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True,
                                     show_scalar_bar=False, specular=0.3)
        if text is not None:
            pl.remove_actor(text, render=False)
        frame = int(name[len("frame_"):-len(".ply")])
        t = f"   t = {times[frame]:.3f}" if frame in times else ""
        text = pl.add_text(f"{sim}   step {n}/{len(files) - 1}{t}", position="upper_left",
                           font_size=TITLE_FONT_SIZE, color="black")
        pl.camera.focal_point = center
        pl.camera.position = center + direction * radius * CAMERA_DISTANCE
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.camera.view_angle = VIEW_ANGLE_DEG
        pl.camera.clipping_range = (radius * NEAR_CLIP, radius * FAR_CLIP)
        img = pl.screenshot(os.path.join(png_dir, f"frame_{n:05d}.png"), return_img=True)
        if video is None:  # the image size may differ from the window size (HiDPI)
            video = VideoWriter(os.path.join(out_dir, sim + ".mp4"),
                                (img.shape[1] // 2 * 2, img.shape[0] // 2 * 2), fps)
        h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2  # yuv420p needs even sizes
        video.write(np.ascontiguousarray(img[:h, :w, :3]))
    pl.close()
    video.close()
    return sim, len(files)


def render_all(frame_root, out_root, jobs, fps):
    sims = sorted(d for d in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, d)))
    print(f"[render] {len(sims)} simulation(s) with {jobs} worker(s)", flush=True)
    errors = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(render_sim, os.path.join(frame_root, s), os.path.join(out_root, s),
                               fps): s for s in sims}
        for fut in as_completed(futures):
            try:
                sim, n = fut.result()
                print(f"[render] {sim}: {n} frames", flush=True)
            except Exception as e:  # keep rendering the others
                errors.append(futures[fut])
                print(f"[render] {futures[fut]}: FAILED ({e})", flush=True)
    return sims, errors


# ----------------------------------------------------------------------------


def ensure_pyvista(out_root):
    try:
        import numpy  # noqa: F401
        import pyvista  # noqa: F401
        return
    except ImportError:
        for root in (out_root, DEFAULT_OUT):
            venv_python = os.path.join(root, ".venv", "bin", "python")
            if os.path.exists(venv_python) and sys.prefix != os.path.join(root, ".venv"):
                os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])
        sys.exit("numpy and pyvista are required for rendering (pip install numpy pyvista)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("--build-dir", default=os.path.join(REPO, "build"))
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output root (default {DEFAULT_OUT})")
    ap.add_argument("--filter", default=DEFAULT_FILTER, help="gtest filter")
    ap.add_argument("--shards", type=int, default=4, help="parallel test processes")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel build and render workers")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-tests", action="store_true", help="render existing <out>/frames")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--keep-frames", action="store_true", help="keep the per-step PLYs")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    frame_root = os.path.join(out_root, "frames")
    os.makedirs(out_root, exist_ok=True)
    if not args.skip_render:
        ensure_pyvista(out_root)
    start = time.time()
    if not args.skip_build:
        build(args.build_dir, args.jobs)
    failed = []
    if not args.skip_tests:
        failed = run_tests(args.build_dir, args.filter, args.shards, frame_root,
                           os.path.join(out_root, "logs"))
    errors = []
    if not args.skip_render:
        _, errors = render_all(frame_root, out_root, args.jobs, args.fps)
        if not args.keep_frames and not errors:
            shutil.rmtree(frame_root, ignore_errors=True)
    print(f"[done] {time.time() - start:.0f} s; output in {out_root}", flush=True)
    return 1 if failed or errors else 0


if __name__ == "__main__":
    sys.exit(main())
