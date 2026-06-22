"""
main.py
Entry point for the Kimera-inspired Geometric World Model pipeline.

Architecture:
  Stream A: RGB-D + Pose → WorkspaceGeometryStream (TSDF, environment only)
  Stream B: RGB → SimulatedSegmentor → ObjectTracker → ObjectGeometryStream
  DSG:      WorkspaceGeometry + Object3D[] → DynamicSceneGraph → JSON output

Usage:
    python main.py --dataset /path/to/tum/sequence --sequence fr1 --frames 100

Output files (in --output_dir):
    world_model_final.json   — full DSG world model
    workspace_mesh.ply       — workspace mesh (if --export_mesh)
    objects/                 — per-object point clouds (if --export_clouds)
"""

import argparse
import os
import sys
import json
import numpy as np

from tum_loader import TUMLoader
from world_model import GeometricWorldModel, WorldModelOutput


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Kimera-inspired Geometric World Model — TUM RGB-D"
    )
    p.add_argument("--dataset",    type=str, required=True,
                   help="Path to TUM sequence directory")
    p.add_argument("--sequence",   type=str, default="fr1",
                   choices=["fr1", "fr2", "fr3"],
                   help="TUM sequence family (default: fr1)")
    p.add_argument("--frames",     type=int, default=100,
                   help="Max frames to process, -1 = all (default: 100)")
    p.add_argument("--output_dir", type=str, default="output",
                   help="Output directory (default: ./output)")
    p.add_argument("--tsdf_voxel", type=float, default=0.025,
                   help="Workspace TSDF voxel size in metres (default: 0.025)")
    p.add_argument("--depth_max",  type=float, default=3.0,
                   help="Max depth in metres (default: 3.0)")
    p.add_argument("--export_mesh",   action="store_true",
                   help="Export workspace mesh as workspace_mesh.ply")
    p.add_argument("--export_clouds", action="store_true",
                   help="Export per-object point clouds to objects/")
    p.add_argument("--no_tsdf",    action="store_true",
                   help="Skip workspace TSDF integration (Stream B only)")
    return p.parse_args()


# ── printing ──────────────────────────────────────────────────────────────────

def print_summary(output: WorldModelOutput) -> None:
    sg   = output.scene_graph
    ws   = sg.get("layer1_workspace") or {}
    objs = sg.get("layer2_objects", [])
    rels = sg.get("layer3_relations", [])

    print("\n" + "=" * 65)
    print(f"  WORLD MODEL  |  frame={output.frame_id}  |  {output.processing_time_ms:.1f} ms")
    print("=" * 65)

    # Layer 1
    print("\n[Layer 1 — Workspace]")
    if ws:
        bmin = [round(v, 3) for v in ws.get("bounds_min", [])]
        bmax = [round(v, 3) for v in ws.get("bounds_max", [])]
        nsrf = len(ws.get("support_surfaces", []))
        print(f"  bounds_min      : {bmin}")
        print(f"  bounds_max      : {bmax}")
        print(f"  support_surfaces: {nsrf}")
        print(f"  frames_integrated: {ws.get('frames_integrated', 0)}")
        for i, surf in enumerate(ws.get("support_surfaces", [])):
            print(f"    surface {i}: height={surf['height']:.3f}m  area={surf['area']:.3f}m²")
    else:
        print("  (not yet available)")

    # Layer 2
    print(f"\n[Layer 2 — Objects]  ({len(objs)} tracked)")
    for obj in objs:
        print(f"\n  [{obj['class_name'].upper()}]  id={obj['object_id']}  "
              f"obs={obj['observations']}  conf={obj['confidence']:.2f}")
        print(f"    centroid          : {[round(v,3) for v in obj['centroid']]}")
        print(f"    extent            : {[round(v,3) for v in obj['extent']]}")
        print(f"    volume_convex_hull: {obj['volume_convex_hull']:.6f} m³")
        print(f"    volume_alpha_shape: {obj['volume_alpha_shape']:.6f} m³")
        print(f"    volume_estimate   : {obj['volume_estimate']:.6f} m³")
        print(f"    visible_ratio     : {obj['visible_ratio']:.3f}")
        print(f"    occluded_by       : {obj['occluded_by']}")
        print(f"    occlusion_type    : {obj['occlusion_type']}")
        print(f"    resting_on        : {obj['resting_on']}")
        acc = obj.get("directional_accessibility", {})
        if acc:
            print(f"    accessibility     : { {k: round(v,2) for k,v in acc.items()} }")

    # Layer 3
    print(f"\n[Layer 3 — Relations]  ({len(rels)} edges)")
    counts: dict = {}
    for e in rels:
        counts[e["relation"]] = counts.get(e["relation"], 0) + 1
    for rel, cnt in sorted(counts.items()):
        print(f"    {rel:<20} {cnt}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n[TUMLoader] sequence={args.sequence}  path={args.dataset}")
    loader = TUMLoader(
        dataset_path=args.dataset,
        sequence=args.sequence,
        max_frames=args.frames,
    )
    print(f"[TUMLoader] {len(loader)} frames loaded")
    if len(loader) == 0:
        print("ERROR: No frames loaded.  Check path and groundtruth.txt.")
        sys.exit(1)

    intrinsic_o3d = loader.get_o3d_intrinsic()
    K             = loader.K

    # ── Build world model ─────────────────────────────────────────────────────
    wm = GeometricWorldModel(
        sequence=args.sequence,
        tsdf_voxel_size=args.tsdf_voxel,
        depth_max=args.depth_max,
        obj_voxel_downsample=0.004,
        obj_min_points=60,
        integrate_workspace=not args.no_tsdf,
    )
    wm.set_intrinsics(intrinsic_o3d, K)

    # ── Per-frame processing ──────────────────────────────────────────────────
    print(f"\n[Pipeline] Processing {len(loader)} frames ...\n")
    last_output = None
    for i, frame in enumerate(loader):
        output     = wm.process_frame(
            rgb=frame.rgb,
            depth=frame.depth,
            pose=frame.pose,
            timestamp=frame.timestamp,
        )
        last_output = output

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  frame {i+1:4d}/{len(loader)}"
                  f"  |  objects={output.n_objects}"
                  f"  |  edges={output.n_edges}"
                  f"  |  {output.processing_time_ms:.1f} ms")

    # ── Final extraction ──────────────────────────────────────────────────────
    print("\n[WorldModel] Extracting final world model ...")
    final = wm.extract_final()
    print_summary(final)

    # ── Export ────────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "world_model_final.json")
    wm.export_json(final, json_path)
    print(f"[Export] World model JSON → {json_path}")

    if args.export_mesh and not args.no_tsdf:
        mesh_path = os.path.join(args.output_dir, "workspace_mesh.ply")
        print(f"[Export] Workspace mesh    → {mesh_path}")
        wm.export_workspace_mesh(mesh_path)

    if args.export_clouds:
        clouds_dir = os.path.join(args.output_dir, "objects")
        print(f"[Export] Object clouds     → {clouds_dir}/")
        wm.export_object_clouds(clouds_dir)

    print("\n[Done]")


if __name__ == "__main__":
    main()