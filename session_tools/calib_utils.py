"""Shared helpers for hand-eye calibration (python3 + cv2 + numpy)."""
import math
import numpy as np
import cv2

DICT_NAME = "DICT_4X4_50"
SQUARES = (5, 7)


def make_board(square_m, marker_m, dict_name=DICT_NAME, squares=SQUARES):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.CharucoBoard(tuple(squares), float(square_m), float(marker_m), d)


def detect_charuco(board, bgr, K, dist, min_corners=6):
    """Detect the board and solvePnP -> dict with rvec/tvec (cam<-target) or None."""
    det = cv2.aruco.CharucoDetector(board)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ch_c, ch_i, _, _ = det.detectBoard(gray)
    if ch_i is None or len(ch_i) < min_corners:
        return None
    obj, img = board.matchImagePoints(ch_c, ch_i)
    if obj is None or len(obj) < min_corners:
        return None
    K = np.asarray(K, float); dist = np.asarray(dist, float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2).sum(1)).mean())
    return {"rvec": rvec.flatten().tolist(), "tvec": tvec.flatten().tolist(),
            "n_corners": int(len(ch_i)), "reproj_px": err}


# ----- transforms (all 4x4 homogeneous) -----
def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(tvec, float).reshape(3)
    return T


def quat_to_R(q):  # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def posquat_to_T(pos, quat_wxyz):
    T = np.eye(4); T[:3, :3] = quat_to_R(quat_wxyz); T[:3, 3] = np.asarray(pos, float)
    return T


def T_inv(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti


def R_to_quat(R):  # wxyz
    w = math.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2; w = max(w, 1e-9)
    return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
            (R[1, 0] - R[0, 1]) / (4 * w)]


def R_to_rpy_deg(R):
    return [math.degrees(a) for a in (math.atan2(R[2, 1], R[2, 2]),
            math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])),
            math.atan2(R[1, 0], R[0, 0]))]


def avg_pose(Ts):
    """Average a list of 4x4 poses (mean translation, normalized mean quat)."""
    tb = np.array([T[:3, 3] for T in Ts])
    qs = np.array([R_to_quat(T[:3, :3]) for T in Ts])
    qs = qs * np.sign(qs[:, 0:1])                 # hemisphere-align before mean
    qm = qs.mean(0); qm /= np.linalg.norm(qm)
    out = np.eye(4); out[:3, :3] = quat_to_R(qm); out[:3, 3] = tb.mean(0)
    spread_mm = float(np.linalg.norm(tb - tb.mean(0), axis=1).mean() * 1000)
    return out, spread_mm
