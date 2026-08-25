#!/usr/bin/env python3
"""Generate hemispherical hand-eye calibration viewpoints around a target point.

The board sits (flat) at --target in base_link. We place the TCP on a hemisphere
above it with the TCP +Z (approach axis) pointing back AT the target, so the
arm-mounted D405 -- which looks roughly along the approach axis -- frames the
board. We add per-pose ROLL about the optical axis so the hand-eye solve gets
rotation diversity about all axes (pure tilt is not enough).

Each candidate is run through cuRobo IK (planner plan_pose); only reachable,
collision-free poses are kept, and we store the JOINT CONFIG at the pose so you
can step through them with move_to_q.py.

Needs the cuRobo planner server up on 127.0.0.1:9997.

Usage:
  python3 session_tools/gen_calib_poses.py --target 0.35,0.08,0.0
  python3 session_tools/gen_calib_poses.py --target 0.35,0.08,0.0 --radii 0.18,0.22,0.26
"""
import argparse, json, math, time, os
import numpy as np
from perturb_loop import PlannerClient

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="0.35,0.08,0.0", help="board centre x,y,z (m) in base_link")
ap.add_argument("--radii", default="0.2", help="camera distances from target / hemisphere radius (m)")
ap.add_argument("--alt", default="45,90", help="altitude band above horizontal (deg); 90=overhead, 45=45deg up. tilt-from-vertical = 90-alt")
ap.add_argument("--n-alt", type=int, default=3, help="altitude rings across the band")
ap.add_argument("--az-center", type=float, default=180.0, help="azimuth centre (deg); 180 = base (-X) side of the board")
ap.add_argument("--az-span", type=float, default=120.0, help="azimuth sector width (deg) centred on --az-center")
ap.add_argument("--roll-range", default="-50,50", help="optical-axis roll sweep (deg), ramped monotonically to avoid wrist flips")
ap.add_argument("--n-az", type=int, default=6, help="azimuth samples per altitude ring")
ap.add_argument("--max-jump", type=float, default=110.0,
                help="reject a pose if any joint moves more than this (deg) from the previous accepted pose "
                     "(blocks ~180-300deg wrist-branch flips; allows normal 90-100deg moves)")
ap.add_argument("--seed-deg", default="0.11,-17.32,110.67,-3.51,-90.10,0.11",
                help="IK seed / capture start config in URDF deg (the robot's base pose)")
ap.add_argument("--out", default=None, help="output json (default captures/calib_poses_<stamp>.json)")
a = ap.parse_args()

p = np.array([float(v) for v in a.target.split(",")])
radii = [float(v) for v in a.radii.split(",")]
alt_lo, alt_hi = (float(v) for v in a.alt.split(","))          # altitude band above horizontal
# tilt-from-vertical rings = 90 - altitude; high altitude (overhead) first -> small first move
alts = [alt_hi - (alt_hi - alt_lo) * i / max(a.n_alt - 1, 1) for i in range(a.n_alt)]
tilts = [math.radians(90.0 - alt) for alt in alts]
# seed/start = the robot's actual base pose (URDF deg -> rad). Capture starts here too.
SEED = list(np.radians([float(v) for v in a.seed_deg.split(",")]))

def R2q(R):
    w = math.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2; w = max(w, 1e-9)
    return [w, (R[2,1]-R[1,2])/(4*w), (R[0,2]-R[2,0])/(4*w), (R[1,0]-R[0,1])/(4*w)]

def lookat_pose(radius, tilt, az, roll):
    """TCP pose viewing target p from a hemisphere point. +Z -> target, rolled."""
    d = np.array([math.sin(tilt)*math.cos(az), math.sin(tilt)*math.sin(az), math.cos(tilt)])
    pos = p + radius * d
    z = -d / np.linalg.norm(d)                      # approach axis: look at target
    up = np.array([1.,0,0]) if abs(z[2]) > 0.95 else np.array([0,0,1.])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll)           # roll about optical axis
    xr, yr = c*x + s*y, -s*x + c*y
    R = np.column_stack([xr, yr, z])
    return pos, R, [*pos, *R2q(R)]

# ---- build candidate viewpoints: near-vertical tool, roll RAMPED monotonically ----
# With the tool within ~15deg of vertical, rotation diversity for hand-eye comes from the
# optical-axis roll. Ramp roll linearly across the whole sequence (not cycled) so the wrist
# (j6) turns gradually and never has to flip between consecutive shots.
cands = []   # (name, tilt_deg, az_deg, roll_deg, radius, goal, R)
roll_lo, roll_hi = (float(v) for v in a.roll_range.split(","))
az_c, az_s = math.radians(a.az_center), math.radians(a.az_span)
total = len(tilts) * a.n_az
k = 0
for ti, tilt in enumerate(tilts):
    r = radii[ti % len(radii)]
    az_order = range(a.n_az) if ti % 2 == 0 else list(reversed(range(a.n_az)))  # snake across the sector
    for j in az_order:
        frac = j / max(a.n_az - 1, 1)                       # 0..1 across the azimuth sector
        az = az_c - az_s / 2 + az_s * frac
        roll = roll_lo + (roll_hi - roll_lo) * k / max(total - 1, 1)   # monotonic ramp
        k += 1
        pos, R, g = lookat_pose(r, tilt, az, math.radians(roll))
        cands.append((f"alt{int(90-math.degrees(tilt))}_az{int(math.degrees(az))}_r{int(roll)}",
                      math.degrees(tilt), math.degrees(az), round(roll, 1), r, g, R))

print(f"target={p.tolist()}  radii={radii}  candidates={len(cands)}  max-jump={a.max_jump}deg")
print("connecting to planner 127.0.0.1:9997 ...")
pc = PlannerClient()
print("planner backend:", pc.rpc({'type':'ping'}).get('backend'))

# chain-seed: each IK seeded from the PREVIOUS accepted config -> one IK branch, small moves.
kept = []
seed = list(SEED)
for name, tilt, az, roll, r, goal, R in cands:
    res = pc.plan_pose(seed, goal)
    if not res.get("success"):
        print(f"  {name:22s} unreachable  status={res.get('status')}")
        continue
    q = list(map(float, res["trajectory"][-1]))               # joint config at the pose
    jump = float(np.degrees(np.abs(np.array(q) - np.array(seed))).max())
    # Gate only jumps BETWEEN viewing poses (kept non-empty). The first home->viewpoint
    # move is naturally large (stow -> over-board) and is a single clean trajectory.
    if kept and jump > a.max_jump:
        print(f"  {name:22s} SKIP big jump {jump:5.0f}deg (IK branch flip vs previous)")
        continue
    kept.append({"name": name, "q_rad": q, "tcp_pos": goal[:3], "tcp_quat": goal[3:],
                 "z_axis": R[:,2].tolist(), "tilt_deg": tilt, "az_deg": az,
                 "roll_deg": roll, "radius": r, "jump_deg": round(jump, 1)})
    seed = q                                                   # advance the chain
    print(f"  {name:22s} OK   jump {jump:5.0f}deg")

# ---- rotation-diversity report ----
if len(kept) >= 2:
    Z = np.array([k["z_axis"] for k in kept])
    ang = np.degrees(np.arccos(np.clip(Z @ Z.T, -1, 1)))
    rolls = np.array([k["roll_deg"] for k in kept])
    jumps = [k["jump_deg"] for k in kept]
    print(f"\nKEPT {len(kept)}/{len(cands)} reachable")
    print(f"  approach-axis spread: max pairwise {ang.max():.0f} deg, mean {ang[ang>0].mean():.0f} deg")
    print(f"  roll coverage: {sorted(set(int(x) for x in rolls))} deg")
    print(f"  inter-pose joint travel: max {max(jumps):.0f} deg, mean {sum(jumps)/len(jumps):.0f} deg (all <= {a.max_jump:.0f})")
    if ang.max() < 30: print("  !! WARNING: low approach-axis diversity -- widen --tilts")
else:
    print(f"\nKEPT {len(kept)}/{len(cands)} -- too few reachable; loosen target/radii/tilts")

stamp = time.strftime("%Y%m%d_%H%M%S")
out = a.out or f"captures/calib_poses_{stamp}.json"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
json.dump({"target": p.tolist(), "poses": kept}, open(out, "w"), indent=2)
print(f"\nsaved {len(kept)} poses -> {out}")
print("step through them with:")
print(f'  for q in $(jq -c ".poses[].q_rad" {out}); do python3 session_tools/move_to_q.py --target "$q" --duration 3; done')
