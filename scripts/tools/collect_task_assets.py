#!/usr/bin/env python3

"""
Collect all file assets actually instantiated by an Isaac Lab training task.

Designed for:
    Isaac Sim 5.1
    Isaac Lab 2.3.x
    ManagerBasedRLEnv

Example:

python scripts/tools/collect_task_assets.py \
    --task Flat-VQR-Wheel-Yaw \
    --num_envs 1 \
    --output /tmp/vqrwheel_yaw_assets \
    --headless

Output:

/tmp/vqrwheel_yaw_assets/
├── _runtime_stage.usda
├── asset_manifest.txt
├── remote_assets.txt
└── collected/
"""

# ============================================================
# 0. Standard imports
# ============================================================

import argparse
import asyncio
import os
import sys
import time


# ============================================================
# 1. Isaac Lab AppLauncher
#
# IMPORTANT:
# AppLauncher must run before importing modules that require
# Isaac Sim / Kit.
# ============================================================

from isaaclab.app import AppLauncher


# ============================================================
# 2. CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="Collect assets used by one Isaac Lab training task."
)

parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Gym task ID, e.g. Flat-VQR-Wheel-Yaw",
)

parser.add_argument(
    "--output",
    type=str,
    default="/tmp/isaac_task_assets",
    help="Output directory.",
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Only 1 env is normally required for asset collection.",
)

parser.add_argument(
    "--warmup_steps",
    type=int,
    default=5,
    help="Simulation steps before exporting the runtime stage.",
)

parser.add_argument(
    "--expect_robot",
    type=str,
    default="VQRWheel",
    help=(
        "Safety check: expected substring in robot USD path. "
        "Use empty string to disable."
    ),
)

# Add:
# --headless
# --device
# etc.
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# This utility is intended to work without GUI.
args_cli.headless = True


# ============================================================
# 3. Launch Isaac Sim
# ============================================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================
# 4. Imports requiring Isaac Sim
# ============================================================

import gymnasium as gym
import torch

import omni.kit.app
import omni.usd

# Register rl_training Gym environments.
import rl_training.tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


# ============================================================
# Helpers
# ============================================================


def print_header(title: str):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def get_robot_usd_path(env_cfg):
    """Return robot USD path if this config uses UsdFileCfg."""

    try:
        return str(env_cfg.scene.robot.spawn.usd_path)
    except Exception:
        return "<unable to resolve robot USD path>"


def print_config_summary(env_cfg):
    """Print important resolved config information."""

    print_header("RESOLVED TRAINING CONFIG")

    robot_path = get_robot_usd_path(env_cfg)

    print(f"Task          : {args_cli.task}")
    print(f"Config class  : {type(env_cfg).__module__}.{type(env_cfg).__name__}")
    print(f"Robot USD     : {robot_path}")

    try:
        print(f"Terrain type  : {env_cfg.scene.terrain.terrain_type}")
    except Exception:
        print("Terrain type  : <not available>")

    try:
        print(f"Terrain gen   : {env_cfg.scene.terrain.terrain_generator}")
    except Exception:
        pass

    try:
        print(f"Terrain USD   : {env_cfg.scene.terrain.usd_path}")
    except Exception:
        pass

    print("=" * 90)

    return robot_path


def export_runtime_root_layer(output_file: str):
    """
    Export the active stage's root layer.

    This intentionally does NOT use:
        omni.usd.get_context().save_as_stage(...)

    save_as_stage goes through the Kit Save-As pipeline and may wait
    on unresolved remote dependencies.

    RootLayer.Export() serializes the authored layer directly.
    """

    print_header("[3/5] EXPORTING RUNTIME ROOT LAYER")

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError(
            "No active USD stage exists."
        )

    root_layer = stage.GetRootLayer()

    if root_layer is None:
        raise RuntimeError(
            "Active USD stage has no root layer."
        )

    print(f"[INFO] Root layer identifier:")
    print(f"       {root_layer.identifier}")

    print()
    print("[INFO] Current layer stack:")

    try:
        for layer in stage.GetLayerStack():
            print(f"       {layer.identifier}")
    except Exception as exc:
        print(
            f"[WARN] Unable to enumerate layer stack: {exc}"
        )

    print()
    print(f"[INFO] Export destination:")
    print(f"       {output_file}")

    start = time.time()

    success = root_layer.Export(output_file)

    elapsed = time.time() - start

    if not success:
        raise RuntimeError(
            f"RootLayer.Export() failed: {output_file}"
        )

    if not os.path.isfile(output_file):
        raise RuntimeError(
            f"Export returned success but file does not exist: "
            f"{output_file}"
        )

    size = os.path.getsize(output_file)

    print(
        f"[INFO] Root layer exported in {elapsed:.3f} s"
    )
    print(
        f"[INFO] Runtime stage size: "
        f"{size / 1024.0 / 1024.0:.3f} MiB"
    )


def enable_collector_extension():
    """Enable omni.kit.tool.collect."""

    print_header("[4/5] LOADING COLLECTOR EXTENSION")

    app = omni.kit.app.get_app()

    ext_manager = app.get_extension_manager()

    extension_name = "omni.kit.tool.collect"

    print(
        f"[INFO] Enabling extension: "
        f"{extension_name}"
    )

    ext_manager.set_extension_enabled_immediate(
        extension_name,
        True,
    )

    # Allow Kit to initialize extension.
    for _ in range(20):
        simulation_app.update()

    print("[INFO] Collector extension loaded.")


def collect_dependencies(
    runtime_stage: str,
    collected_dir: str,
):
    """
    Schedule Collector.collect() on Kit's event loop.

    DO NOT use asyncio.run().
    DO NOT call loop.run_until_complete().

    Instead:
        asyncio.ensure_future(...)
        simulation_app.update()

    Kit advances its asyncio tasks as its application loop updates.
    """

    print_header("[5/5] COLLECTING ASSET DEPENDENCIES")

    from omni.kit.tool.collect import Collector

    os.makedirs(
        collected_dir,
        exist_ok=True,
    )

    print(f"[INFO] Source stage:")
    print(f"       {runtime_stage}")

    print(f"[INFO] Destination:")
    print(f"       {collected_dir}")

    print()
    print(
        "[INFO] usd_only=False -> include USD, MDL, "
        "textures, HDR, etc."
    )

    print(
        "[INFO] flat_collection=False -> preserve "
        "dependency directory structure."
    )

    collector = Collector(
        usd_path=runtime_stage,
        collect_dir=collected_dir,

        # Collect ALL supported dependency types.
        usd_only=False,

        # Keep meaningful directory layout.
        flat_collection=False,

        # Do not restrict collection to materials.
        material_only=False,

        # Useful when re-running the script.
        skip_existing=True,

        # Conservative value for remote S3 access.
        max_concurrent_tasks=8,

        # Collect everything, not just default prim.
        default_prim_only=False,
    )

    last_progress = {
        "current": -1,
        "total": -1,
    }

    def progress_callback(current, total):

        # Avoid flooding stdout if callback repeats.
        if (
            current != last_progress["current"]
            or total != last_progress["total"]
        ):
            print(
                f"\r[COLLECT] {current}/{total}",
                end="",
                flush=True,
            )

            last_progress["current"] = current
            last_progress["total"] = total

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Schedule onto Kit's asyncio loop.
    #
    # Do not:
    #
    # asyncio.run(...)
    #
    # and do not:
    #
    # loop.run_until_complete(...)
    #
    # --------------------------------------------------------

    task = asyncio.ensure_future(
        collector.collect(
            progress_callback=progress_callback,
        )
    )

    print()
    print("[INFO] Collector task scheduled.")

    start = time.time()
    last_status_time = start

    try:

        while not task.done():

            # This advances:
            #
            # - Kit
            # - omni.client
            # - asyncio jobs
            # - collector copy jobs
            #
            simulation_app.update()

            now = time.time()

            # Heartbeat every 10 seconds.
            if now - last_status_time >= 10.0:

                elapsed = now - start

                try:
                    status = collector.get_status()
                except Exception:
                    status = "<unavailable>"

                print()
                print(
                    f"[INFO] Collector still running "
                    f"({elapsed:.1f}s)"
                )
                print(
                    f"[INFO] Status: {status}"
                )

                last_status_time = now

        print()

        # Propagates exception raised inside collector.
        success, collected_stage = task.result()

    except KeyboardInterrupt:

        print()
        print("[WARN] Ctrl+C received. Cancelling collector...")

        try:
            collector.cancel()
        except Exception:
            pass

        raise

    if not success:

        try:
            status = collector.get_status()
        except Exception:
            status = "<unavailable>"

        raise RuntimeError(
            "Collector failed.\n"
            f"Status: {status}"
        )

    mapping = (
        collector.get_source_target_url_mapping()
    )

    elapsed = time.time() - start

    print(
        f"[INFO] Asset collection completed "
        f"in {elapsed:.1f}s"
    )

    print(
        f"[INFO] Collected root stage:"
    )
    print(
        f"       {collected_stage}"
    )

    return collector, mapping, collected_stage


def save_manifest(
    output_dir: str,
    mapping: dict,
):
    """Save complete and remote-only asset manifests."""

    manifest_path = os.path.join(
        output_dir,
        "asset_manifest.txt",
    )

    remote_manifest_path = os.path.join(
        output_dir,
        "remote_assets.txt",
    )

    items = sorted(
        mapping.items(),
        key=lambda x: str(x[0]),
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        for src, dst in items:

            f.write(f"{src}\n")
            f.write(f"  -> {dst}\n")
            f.write("\n")

    remote_count = 0

    with open(
        remote_manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        for src, dst in items:

            src_str = str(src)

            if src_str.startswith(
                (
                    "http://",
                    "https://",
                    "omniverse://",
                    "s3://",
                )
            ):

                remote_count += 1

                f.write(f"{src}\n")
                f.write(f"  -> {dst}\n")
                f.write("\n")

    print()
    print("[INFO] Manifest:")
    print(f"       {manifest_path}")

    print("[INFO] Remote-only manifest:")
    print(f"       {remote_manifest_path}")

    print(
        f"[INFO] Total mappings: {len(items)}"
    )

    print(
        f"[INFO] Remote assets : {remote_count}"
    )

    return manifest_path, remote_manifest_path


# ============================================================
# Main
# ============================================================


def main():

    print_header("ISAAC LAB TRAINING TASK ASSET COLLECTOR")

    output_dir = os.path.abspath(
        os.path.expanduser(args_cli.output)
    )

    collected_dir = os.path.join(
        output_dir,
        "collected",
    )

    runtime_stage = os.path.join(
        output_dir,
        "_runtime_stage.usda",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print(f"Task        : {args_cli.task}")
    print(f"Output      : {output_dir}")
    print(f"Num envs    : {args_cli.num_envs}")
    print(f"Warmup      : {args_cli.warmup_steps} steps")
    print(f"Device      : {args_cli.device}")

    # ========================================================
    # 1/5 Parse exact resolved environment config
    # ========================================================

    print_header("[1/5] PARSING ENVIRONMENT CONFIG")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    robot_usd_path = print_config_summary(
        env_cfg
    )

    # --------------------------------------------------------
    # Guard against accidentally collecting Lite3 again.
    # --------------------------------------------------------

    if args_cli.expect_robot:

        if args_cli.expect_robot not in robot_usd_path:

            raise RuntimeError(
                "\nWRONG ROBOT CONFIG DETECTED\n"
                "---------------------------\n"
                f"Expected robot : {args_cli.expect_robot}\n"
                f"Actual USD     : {robot_usd_path}\n"
                f"Task           : {args_cli.task}\n"
                "\n"
                "Aborting before environment creation."
            )

    # ========================================================
    # 2/5 Create the actual ManagerBasedRLEnv
    #
    # IMPORTANT:
    # This is synchronous.
    #
    # Isaac Lab asset checks may internally use:
    # loop.run_until_complete(...)
    #
    # Therefore gym.make() MUST NOT be inside async def.
    # ========================================================

    print_header("[2/5] CREATING ENVIRONMENT")

    print("[INFO] Calling gym.make()...")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    base_env = env.unwrapped

    print("[INFO] Environment created.")

    print("[INFO] Resetting environment...")

    env.reset()

    # --------------------------------------------------------
    # Run a few real simulation steps.
    #
    # This allows:
    # - debug visualizers
    # - markers
    # - sensors
    # - runtime-created prims
    # - lazy USD assets
    #
    # to be instantiated.
    # --------------------------------------------------------

    action_dim = (
        base_env.action_manager.total_action_dim
    )

    num_envs = base_env.num_envs

    actions = torch.zeros(
        (
            num_envs,
            action_dim,
        ),
        dtype=torch.float32,
        device=base_env.device,
    )

    print(
        f"[INFO] Warm-up: "
        f"{args_cli.warmup_steps} simulation steps"
    )

    for i in range(args_cli.warmup_steps):

        env.step(actions)

        # Let Kit process asset requests.
        simulation_app.update()

        print(
            f"\r[INFO] Warm-up step "
            f"{i + 1}/{args_cli.warmup_steps}",
            end="",
            flush=True,
        )

    print()

    # Extra Kit frames for deferred marker creation/loading.
    print(
        "[INFO] Processing deferred Kit operations..."
    )

    for _ in range(20):
        simulation_app.update()

    print("[INFO] Runtime scene initialized.")

    # ========================================================
    # 3/5 Export stage root layer
    # ========================================================

    export_runtime_root_layer(
        runtime_stage
    )

    # ========================================================
    # 4/5 Enable collector
    # ========================================================

    enable_collector_extension()

    # ========================================================
    # 5/5 Collect dependencies
    # ========================================================

    collector = None

    try:

        (
            collector,
            mapping,
            collected_stage,
        ) = collect_dependencies(
            runtime_stage,
            collected_dir,
        )

        manifest, remote_manifest = save_manifest(
            output_dir,
            mapping,
        )

        # ====================================================
        # Final summary
        # ====================================================

        print_header("COLLECTION COMPLETE")

        print(f"Task:")
        print(f"  {args_cli.task}")

        print()
        print("Robot:")
        print(f"  {robot_usd_path}")

        print()
        print("Runtime stage:")
        print(f"  {runtime_stage}")

        print()
        print("Collected stage:")
        print(f"  {collected_stage}")

        print()
        print("Collected directory:")
        print(f"  {collected_dir}")

        print()
        print("Manifest:")
        print(f"  {manifest}")

        print()
        print("Remote assets:")
        print(f"  {remote_manifest}")

        print()
        print(
            f"Total mapped files: "
            f"{len(mapping)}"
        )

        print()
        print(
            "Next commands:"
        )

        print(
            f"\n  du -sh {collected_dir}"
        )

        print(
            f"\n  cat {remote_manifest}"
        )

        print(
            "\n  grep -Ei "
            "'amazonaws|omniverse|https?://' "
            f"{remote_manifest}"
        )

        print()

    finally:

        if collector is not None:

            try:
                collector.destroy()
            except Exception:
                pass

        try:
            env.close()
        except Exception:
            pass


# ============================================================
# Entry point
# ============================================================

try:

    main()

except KeyboardInterrupt:

    print()
    print("[INFO] Interrupted by user.")

except Exception as exc:

    print()
    print_header("FAILED")

    print(
        f"{type(exc).__name__}: {exc}"
    )

    raise

finally:

    print()
    print("[INFO] Closing Isaac Sim...")

    simulation_app.close()
