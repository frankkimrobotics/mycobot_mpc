#!/usr/bin/env python3
"""
Probe the robot's HAL (on the Raspi) for mass / inertia and related parameters.

Run on the Raspberry Pi with LinuxCNC (and HAL) running. Uses halcmd to list
params and pins, then filters for mass/inertia/gravity/link/payload and
reports current values.

Usage:
  On Raspi (with LinuxCNC running):
    python3 probe_hal_mass_inertia.py

  Optional: dump full param/pin list to a file for inspection:
    python3 probe_hal_mass_inertia.py --dump hal_dump.txt
"""

import re
import subprocess
import sys
import argparse

HALCMD = "halcmd"

# Substrings that indicate mass/inertia/robot dynamics in HAL names
MASS_INERTIA_KEYWORDS = [
    "mass", "iner", "inertia", "inertial", "gravity", "linkvec", "link.",
    "payload", "loadmass", "tchp",  # pro600 tchp = touch/inertia params
    "baseInertial", "com", "cog", "centerofmass",
]

# Known param/pin name patterns from elerob_*.hal (colli, pro600)
KNOWN_MASS_INERTIA_PATTERNS = [
    "colli.baseInertialParamsBott",
    "colli.baseInertialParamsTop",
    "colli.linkVec",
    "colli.gravity",
    "colli.payloadDiff",
    "pro600.tchp0.",
    "pro600.tchp4.",
    "pro600.tchp.loadmass",
]

# Explicit list to probe when we know the config (elerob_mpc.hal / elerob_invdyn.hal)
# So we report them even if show param/pin parsing misses some
KNOWN_PARAMS_TO_PROBE = [
    "colli.gravity",
    "colli.payloadDiff",
    "colli.linkVec.00", "colli.linkVec.01", "colli.linkVec.02",
    "colli.linkVec.03", "colli.linkVec.04",
    "colli.baseInertialParamsBott.00", "colli.baseInertialParamsTop.00",
]
# Add a few tchp samples (pro600 inertia/touch params)
for i in (0, 4):
    for j in ("00", "22", "36", "37"):
        KNOWN_PARAMS_TO_PROBE.append(f"pro600.tchp{i}.{j}")
KNOWN_PARAMS_TO_PROBE.append("pro600.tchp.loadmass")


def run_halcmd(*args):
    """Run halcmd with given args; return (success, stdout, stderr)."""
    try:
        r = subprocess.run(
            [HALCMD] + list(args),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0, r.stdout or "", r.stderr or ""
    except FileNotFoundError:
        return False, "", f"{HALCMD} not found (run on Raspi with LinuxCNC?)"
    except subprocess.TimeoutExpired:
        return False, "", "halcmd timed out"
    except Exception as e:
        return False, "", str(e)


def get_hal_names_from_show(show_stdout, kind="pin"):
    """Parse 'halcmd show param' or 'show pin' output and return list of HAL names.
    Handles both default and -s (script) output.
    """
    names = []
    for line in show_stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("Component") or line.startswith("Parameters") or line.startswith("Pins"):
            continue
        # Format can be: "id  owner  name  type  value" or "name (type) value" or "name => ..."
        # Name usually contains a dot (comp.pin_or_param)
        tokens = line.split()
        for t in tokens:
            if "." in t and not t.startswith("=>"):
                # Strip trailing punctuation from token
                name = re.sub(r"[,\(\)].*$", "", t)
                if name and "." in name:
                    names.append(name)
                    break
        # Also try: match "comp.name" anywhere
        match = re.search(r"([a-zA-Z0-9_]+\.[a-zA-Z0-9_.]+)", line)
        if match:
            names.append(match.group(1))
    return list(dict.fromkeys(names))  # unique, order preserved


def matches_mass_inertia(name):
    """True if name looks like a mass/inertia/dynamics-related HAL item."""
    lower = name.lower()
    for kw in MASS_INERTIA_KEYWORDS:
        if kw.lower() in lower:
            return True
    for pat in KNOWN_MASS_INERTIA_PATTERNS:
        if pat in name:
            return True
    return False


def get_value(name):
    """Get current value of a HAL pin or parameter via halcmd getp."""
    ok, out, err = run_halcmd("getp", name)
    if not ok:
        return None
    return out.strip()


def main():
    ap = argparse.ArgumentParser(description="Probe Raspi HAL for mass/inertia params")
    ap.add_argument("--dump", type=str, metavar="FILE", help="Dump full show param/pin to FILE")
    ap.add_argument("--verbose", "-v", action="store_true", help="List all HAL names considered")
    args = ap.parse_args()

    print("Probing HAL for mass / inertia / gravity / link / payload...")
    print("(Run this on the Raspi with LinuxCNC and HAL running.)\n")

    # 1. Get all params and pins
    ok_param, out_param, err_param = run_halcmd("show", "param")
    ok_pin, out_pin, err_pin = run_halcmd("show", "pin")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write("=== halcmd show param ===\n")
            f.write(out_param)
            f.write("\n=== halcmd show pin ===\n")
            f.write(out_pin)
        print(f"Full dump written to {args.dump}")

    if not ok_param and not ok_pin:
        print("ERROR: Could not run halcmd. Is LinuxCNC running on this machine?")
        if err_param:
            print("  param:", err_param.strip())
        if err_pin:
            print("  pin:", err_pin.strip())
        sys.exit(1)

    # 2. Collect candidate names from show output
    param_names = get_hal_names_from_show(out_param, "param") if ok_param else []
    pin_names = get_hal_names_from_show(out_pin, "pin") if ok_pin else []

    # 3. Also add known names we might not parse (pro600.tchp0.00 etc.)
    all_names = list(dict.fromkeys(param_names + pin_names))

    # 4. Filter to mass/inertia related (from parsed names + known list)
    related = [n for n in all_names if matches_mass_inertia(n)]
    for known in KNOWN_PARAMS_TO_PROBE:
        if known not in related:
            related.append(known)
    # Sort for stable output: group by prefix (colli., pro600., etc.)
    related.sort(key=lambda x: (x.split(".")[0], x))

    if args.verbose:
        print("HAL names matching mass/inertia keywords or known patterns:")
        for n in related:
            print(" ", n)
        print()

    # 5. Report values
    print("--- Mass / Inertia / Dynamics in HAL ---\n")
    if not related:
        print("No mass/inertia-related params or pins found.")
        print("Keywords used:", ", ".join(MASS_INERTIA_KEYWORDS))
        print("Known patterns:", ", ".join(KNOWN_MASS_INERTIA_PATTERNS))
        sys.exit(0)

    # Group by component for readability
    by_comp = {}
    for name in related:
        comp = name.split(".")[0]
        by_comp.setdefault(comp, []).append(name)

    for comp in sorted(by_comp.keys()):
        names = by_comp[comp]
        print(f"[{comp}]")
        for name in names:
            val = get_value(name)
            if val is not None:
                print(f"  {name} = {val}")
            else:
                print(f"  {name} = (read failed)")
        print()

    # Summary
    print("--- Summary ---")
    print(f"  Total mass/inertia-related items: {len(related)}")
    print("  Components:", ", ".join(sorted(by_comp.keys())))
    print()
    print("Note: colli.baseInertialParams* and pro600.tchp* are used by the")
    print("collision/dynamics stack (collidet_panda). Link lengths: colli.linkVec.*")


if __name__ == "__main__":
    main()
