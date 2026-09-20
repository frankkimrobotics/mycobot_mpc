"""Uniform cubic B-spline reference for the streamed (welded) trajectory -- 2026-09-19.

Drop-in C2 replacement for the piecewise-linear welder in ``robot_hal.py``
(``_weld_chunk`` / ``_sample_ref``).  Same welding semantics, same (q, v, state)
contract plus an analytic acceleration.

Time base
---------
Control point ``i`` of a welded message lives at wall time ``t_anchor + i*dt``
(uniform knots).  The curve on the span ``u in [i, i+1)`` (``u = (t - t0)/dt``)
is the uniform cubic B-spline over ``P[i-1], P[i], P[i+1], P[i+2]``:

    S(u) = ( b0 P[i-1] + b1 P[i] + b2 P[i+1] + b3 P[i+2] ) / 6,   s = u - i
    b0 = (1-s)^3          b1 = 3s^3 - 6s^2 + 4
    b2 = -3s^3 + 3s^2 + 3s + 1                     b3 = s^3

which is *centred*: at a knot (s = 0) the value is (P[i-1] + 4 P[i] + P[i+1])/6.
Derivatives are analytic (differentiate the basis, divide by dt / dt^2), so the
velocity feedforward is exact instead of a finite difference, and v is C1
(a is continuous too -- C2 curve).

Approximation and lag
---------------------
This is an *approximating* spline: it does not interpolate its control points.
At a knot it is off by ``(P[i-1] - 2 P[i] + P[i+1]) / 6`` = one sixth of the
second difference, i.e. ``a * dt^2 / 6`` -- 0.4 % of a 0.1 s step for a smooth
target sequence.  Because the basis is centred, the *phase* lag of the curve
against the control-point sequence is ~0 in the interior, but a knot can only be
smoothed once its two neighbours are known, so the last knot the curve tracks
faithfully is one knot (``dt``) behind the newest control point: the usable
reference ends ~1 knot before the end of the point list.  That is why the
desktop always appends one extrapolated point past the newest target (see
``real_policy_ctrl.PiLink.send_segment``) -- the extrapolated point is the one
that gets "eaten" by the lag, and the live target is tracked in phase.

Ends
----
Out-of-range control points are clamped (the first/last point is repeated), so
the curve is defined over the whole ``[t0, t0 + (n-1)*dt]`` instead of stopping
one span short at each end, and a repeated end point is reached exactly (with
three equal points around a knot the basis sums to that point).  With a single
end point the value at the very end is ``(5 P_end + P_neighbour)/6``; the
streamed messages (and the probe) repeat the tail, so this only affects the
first span of a brand-new stream.

Python 3.7 / pure stdlib (no numpy): the Pi interpreter runs this at 100 Hz.
"""

MAX_JOINTS = 6
MAX_PTS = 600          # ~60 s of 10 Hz history; older control points are trimmed


def _basis(s):
    """(b0..b3) / 6 for position, and the same for d/ds and d2/ds2."""
    om = 1.0 - s
    b = (om * om * om / 6.0,
         (3.0 * s * s * s - 6.0 * s * s + 4.0) / 6.0,
         (-3.0 * s * s * s + 3.0 * s * s + 3.0 * s + 1.0) / 6.0,
         s * s * s / 6.0)
    d = (-0.5 * om * om,
         (3.0 * s * s - 4.0 * s) * 0.5,
         (-3.0 * s * s + 2.0 * s + 1.0) * 0.5,
         0.5 * s * s)
    dd = (om,
          3.0 * s - 2.0,
          1.0 - 3.0 * s,
          s)
    return b, d, dd


class SplineRef(object):
    """Welded uniform cubic B-spline reference q_ref(t).

    Not thread safe: the caller (robot_hal) holds ``_ref_lock`` around weld/sample.
    """

    def __init__(self, ndof=MAX_JOINTS):
        self.ndof = ndof
        self.t0 = 0.0
        self.dt = 0.1
        self.pts = []          # control points, pts[i] at t0 + i*dt
        self.welds = 0

    # ------------------------------------------------------------------ weld
    def clear(self):
        self.pts = []
        self.welds = 0

    def weld(self, t_anchor, dt, pts):
        """Drop control points at t >= t_anchor - 1e-6, append `pts` at t_anchor + i*dt.

        Same semantics as robot_hal._weld_chunk.  If the new anchor does not land
        on the existing uniform grid (or dt changed, or there is a gap), the
        reference restarts at t_anchor -- the desktop re-sends its recent history
        every decision, so a restart costs nothing.
        """
        t_anchor = float(t_anchor)
        dt = float(dt)
        if dt <= 0.0:
            dt = 0.1
        new = [[float(x) for x in p][:self.ndof] for p in pts]
        if not new:
            return len(self.pts)
        keep = False
        if self.pts and abs(dt - self.dt) < 1e-9:
            idx = (t_anchor - self.t0) / dt
            i0 = int(round(idx))
            if abs(idx - i0) < 1e-3 and 0 <= i0 <= len(self.pts):
                del self.pts[i0:]
                self.pts.extend(new)
                keep = True
        if not keep:
            self.t0 = t_anchor
            self.dt = dt
            self.pts = new
        if len(self.pts) > MAX_PTS:                      # trim stale history
            drop = len(self.pts) - MAX_PTS
            del self.pts[:drop]
            self.t0 += drop * self.dt
        self.welds += 1
        return len(self.pts)

    # ---------------------------------------------------------------- sample
    def _p(self, j):
        n = len(self.pts)
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1
        return self.pts[j]

    def _eval(self, u, want_deriv=True):
        n = len(self.pts)
        i = int(u)
        if u < 0.0:
            i = 0
        if i > n - 2:
            i = n - 2
        if i < 0:
            i = 0
        s = u - i
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        b, d, dd = _basis(s)
        pm, p0, p1, p2 = self._p(i - 1), self._p(i), self._p(i + 1), self._p(i + 2)
        nd = self.ndof
        q = [b[0] * pm[k] + b[1] * p0[k] + b[2] * p1[k] + b[3] * p2[k] for k in range(nd)]
        if not want_deriv:
            return q, [0.0] * nd, [0.0] * nd
        inv = 1.0 / self.dt
        v = [(d[0] * pm[k] + d[1] * p0[k] + d[2] * p1[k] + d[3] * p2[k]) * inv for k in range(nd)]
        a = [(dd[0] * pm[k] + dd[1] * p0[k] + dd[2] * p1[k] + dd[3] * p2[k]) * inv * inv for k in range(nd)]
        return q, v, a

    def sample(self, t):
        """-> (q[ndof], v[ndof], a[ndof], state) with state before|in|after|empty."""
        n = len(self.pts)
        nd = self.ndof
        if n == 0:
            return None, None, None, "empty"
        zero = [0.0] * nd
        if n == 1:
            return list(self.pts[0]), list(zero), list(zero), "after"
        u = (t - self.t0) / self.dt
        umax = float(n - 1)
        if u <= 0.0:
            q, _, _ = self._eval(0.0, want_deriv=False)
            return q, list(zero), list(zero), "before"
        if u >= umax:
            q, _, _ = self._eval(umax, want_deriv=False)
            return q, list(zero), list(zero), "after"
        q, v, a = self._eval(u)
        return q, v, a, "in"

    # ------------------------------------------------------------------ misc
    def t_end(self):
        return self.t0 + (len(self.pts) - 1) * self.dt if self.pts else None

    def __len__(self):
        return len(self.pts)
