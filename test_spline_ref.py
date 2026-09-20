#!/usr/bin/env python3
"""fix 3 offline check: linear welder (robot_hal) vs SplineRef on a policy-like target stream.

The two reference builders are driven with exactly the controller's cadence: one weld every
CTRL_DT = 0.1 s, the linear one with the 3-point segment of PiLink.send_segment
([q_prev, q_new, extrapolated]) and the spline one with the 5-point control polygon of
PiLink._send_spline_segment ([last 4 targets, extrapolated], anchor = t - 2*dt).  Both are then
sampled at the 100 Hz control rate and compared.

The linear welder functions are COPIES of robot_hal._weld_chunk / _sample_ref (robot_hal itself
cannot be imported off the Pi: it needs linuxcnc/hal).

Run: python3 test_spline_ref.py
"""
import bisect
import math

from spline_ref import SplineRef

NJ = 6
CTRL_DT = 0.1
LOOP_DT = 0.01


# ------------------------------------------------- copy of robot_hal's linear welder
class LinearRef(object):
    def __init__(self):
        self.t, self.q = [], []

    def weld(self, t_anchor, dt, pts):
        k = len(self.t)
        while k > 0 and self.t[k - 1] >= t_anchor - 1e-6:
            k -= 1
        del self.t[k:]
        del self.q[k:]
        for i, p in enumerate(pts):
            self.t.append(t_anchor + i * dt)
            self.q.append([float(x) for x in p][:NJ])
        return len(self.t)

    def sample(self, t):
        n = len(self.t)
        if n == 0:
            return None, None, "empty"
        if t <= self.t[0]:
            return list(self.q[0]), [0.0] * NJ, "before"
        if t >= self.t[-1]:
            return list(self.q[-1]), [0.0] * NJ, "after"
        i = bisect.bisect_right(self.t, t) - 1
        t0, t1 = self.t[i], self.t[i + 1]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        q0, q1 = self.q[i], self.q[i + 1]
        return ([q0[j] + a * (q1[j] - q0[j]) for j in range(NJ)],
                [(q1[j] - q0[j]) / (t1 - t0) for j in range(NJ)], "in")


# ------------------------------------------------------------------ target streams
def minjerk_targets(total_deg=30.0, T=2.0, dt=CTRL_DT):
    """Targets at 10 Hz from a min-jerk 30 deg move (joint 0; the others stay put)."""
    out = []
    n = int(round(T / dt)) + 1
    for k in range(n):
        s = min(1.0, max(0.0, (k * dt) / T))
        f = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        out.append([total_deg * f] + [0.0] * (NJ - 1))
    for _ in range(5):
        out.append(list(out[-1]))
    return out


def reversal_targets(step=1.0, n=40, dt=CTRL_DT):
    """Constant 1 deg/decision ramp that reverses half way (worst case for kinks)."""
    out, v = [[0.0] * NJ], step
    for k in range(n):
        if k == n // 2:
            v = -step
        prev = out[-1]
        out.append([prev[0] + v] + [0.0] * (NJ - 1))
    return out


# ------------------------------------------------------------------------ drivers
def run_pair(targets, dt=CTRL_DT):
    """Replay the controller: decision k at t = k*dt sends target[k]; sample both refs at 100 Hz."""
    lin, spl = LinearRef(), SplineRef(NJ)
    hist = []
    ts, q_lin, v_lin, q_spl, v_spl, a_spl = [], [], [], [], [], []
    t_first_sample = None
    for k, tgt in enumerate(targets):
        t_dec = k * dt
        prev = hist[-1] if hist else tgt
        # --- linear: [q_prev_target, q_target, extrapolated], anchored at the decision
        ext = [2 * tgt[j] - prev[j] for j in range(NJ)]
        lin.weld(t_dec, dt, [prev, tgt, ext])
        # --- spline: last 4 targets + extrapolation, anchor = t_dec - 2*dt
        hist.append(list(tgt))
        h = hist[-4:]
        while len(h) < 4:
            h.insert(0, list(h[0]))
        sext = [2 * h[-1][j] - h[-2][j] for j in range(NJ)]
        spl.weld(t_dec - 2 * dt, dt, h + [sext])
        # --- 100 Hz control loop over this decision interval
        if t_first_sample is None:
            t_first_sample = t_dec
        t = t_dec
        while t < t_dec + dt - 1e-9:
            ql, vl, _ = lin.sample(t)
            qs, vs, as_, _ = spl.sample(t)
            ts.append(t)
            q_lin.append(ql[0]); v_lin.append(vl[0])
            q_spl.append(qs[0]); v_spl.append(vs[0]); a_spl.append(as_[0])
            t += LOOP_DT
    return ts, q_lin, v_lin, q_spl, v_spl, a_spl


def max_jump(v, idx=None):
    rng = range(len(v) - 1) if idx is None else idx
    return max(abs(v[i + 1] - v[i]) for i in rng)


def target_at(targets, t, dt=CTRL_DT):
    """The reference the linear scheme aims at: target[k] is the value at (k+1)*dt."""
    u = t / dt - 1.0
    if u <= 0:
        return targets[0][0]
    i = int(u)
    if i >= len(targets) - 1:
        return targets[-1][0]
    a = u - i
    return targets[i][0] + a * (targets[i + 1][0] - targets[i][0])


def dev_and_lag(ts, q, targets, dt=CTRL_DT):
    dev = max(abs(q[i] - target_at(targets, ts[i])) for i in range(len(ts)))
    best, best_e = 0.0, None
    sh = -0.10
    while sh <= 0.1001:
        e = 0.0
        for i in range(len(ts)):
            d = q[i] - target_at(targets, ts[i] - sh)
            e += d * d
        if best_e is None or e < best_e:
            best_e, best = e, sh
        sh += 0.002
    return dev, best


def report(name, targets):
    ts, ql, vl, qs, vs, as_ = run_pair(targets)
    jl, js = max_jump(vl), max_jump(vs)
    n_dec = int(round(CTRL_DT / LOOP_DT))
    weld = [i for i in range(len(ts) - 1) if (i + 1) % n_dec == 0]        # sample pairs across a weld
    inner = [i for i in range(len(ts) - 1) if (i + 1) % n_dec != 0]
    jlw, jsw = max_jump(vl, weld), max_jump(vs, weld)
    jli, jsi = max_jump(vl, inner), max_jump(vs, inner)
    dev_l, lag_l = dev_and_lag(ts, ql, targets)
    dev_s, lag_s = dev_and_lag(ts, qs, targets)
    amax_l = max_jump(vl) / LOOP_DT      # linear: acceleration is impulsive, report the finite diff
    amax_s = max(abs(x) for x in as_)
    print("\n=== %s (%d decisions, %d loop samples) ===" % (name, len(targets), len(ts)))
    print("  max |dv| between consecutive 10 ms samples : linear %.4f deg/s   spline %.4f deg/s   (%.0fx smaller)"
          % (jl, js, (jl / js) if js > 0 else float("inf")))
    print("      of which at the weld instant          : linear %.4f            spline %.4f" % (jlw, jsw))
    print("      of which inside a decision             : linear %.4f            spline %.4f" % (jli, jsi))
    print("  max |a| (finite diff / analytic)           : linear %.1f deg/s^2  spline %.1f deg/s^2" % (amax_l, amax_s))
    print("  max |q_ref - target(t)|                    : linear %.4f deg      spline %.4f deg" % (dev_l, dev_s))
    print("  best-fit time shift (lag > 0 = late)       : linear %+.3f s        spline %+.3f s" % (lag_l, lag_s))
    return jl, js, dev_s, lag_s


def test_weld_semantics():
    """The spline welder must drop t >= t_anchor-1e-6 and keep the uniform base (as _weld_chunk)."""
    s = SplineRef(NJ)
    s.weld(100.0, 0.1, [[float(i)] * NJ for i in range(5)])
    assert len(s) == 5
    s.weld(100.3, 0.1, [[9.0] * NJ, [9.5] * NJ])       # drops index 3,4 -> 3 + 2 = 5
    assert len(s) == 5, len(s)
    assert abs(s.pts[3][0] - 9.0) < 1e-9 and abs(s.t0 - 100.0) < 1e-9
    s.weld(200.0, 0.1, [[1.0] * NJ] * 3)               # off grid -> restart
    assert len(s) == 3 and abs(s.t0 - 200.0) < 1e-9
    q, v, a, st = s.sample(199.0)
    assert st == "before" and v == [0.0] * NJ
    q, v, a, st = s.sample(201.0)
    assert st == "after"
    assert s.sample(200.05)[3] == "in"
    assert SplineRef(NJ).sample(1.0)[3] == "empty"
    # C2: the analytic derivatives must match finite differences of the sampled curve
    s2 = SplineRef(NJ)
    s2.weld(0.0, 0.1, [[math.sin(i * 0.3) * 10.0] * NJ for i in range(12)])
    h = 1e-4
    worst_v = worst_a = 0.0
    t = 0.15
    while t < 1.0:
        q0 = s2.sample(t - h)[0][0]; q1 = s2.sample(t + h)[0][0]; qc = s2.sample(t)[0][0]
        worst_v = max(worst_v, abs(s2.sample(t)[1][0] - (q1 - q0) / (2 * h)))
        worst_a = max(worst_a, abs(s2.sample(t)[2][0] - (q1 - 2 * qc + q0) / (h * h)))
        t += 0.013
    print("  analytic vs finite-difference derivatives: max |dv| %.2e deg/s, max |da| %.2e deg/s^2" % (worst_v, worst_a))
    assert worst_v < 1e-6 and worst_a < 1e-2
    # endpoint clamping: a repeated end control point is reached exactly
    s3 = SplineRef(NJ)
    s3.weld(0.0, 0.25, [[0.0] * NJ, [1.0] * NJ, [1.0] * NJ, [1.0] * NJ, [1.0] * NJ])
    assert abs(s3.sample(1.0)[0][0] - 1.0) < 1e-12
    print("  weld semantics, state machine, endpoint clamp: OK")


if __name__ == "__main__":
    print("== weld / derivative / endpoint checks ==")
    test_weld_semantics()
    report("min-jerk 30 deg in 2 s, 10 Hz targets", minjerk_targets())
    report("1 deg/decision ramp with a direction reversal", reversal_targets())
    # timing on this machine (the Pi is ~20x slower, still << 10 ms)
    import time as _t
    s = SplineRef(NJ)
    s.weld(0.0, 0.1, [[float(i % 7)] * NJ for i in range(200)])
    t0 = _t.time()
    for i in range(10000):
        s.sample(5.0 + i * 1e-4)
    print("\nsample(): %.1f us per call (6 dof, 200 control points)" % (1e6 * (_t.time() - t0) / 10000))
