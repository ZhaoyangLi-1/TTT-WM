"""Shared action / pose conversion helpers used by the parquet-image diffusion
policy dataset and by Stage 2.2's video-frame dataset.

Convention note: rotation_6d here matches pytorch3d's
``matrix_to_rotation_6d`` — take the first two ROWS of the rotation matrix
and flatten row-major. Keep this in sync with any rollout-time decoding.
"""

from __future__ import annotations

import numpy as np


def axis_angle_to_rotation_6d(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float32)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, np.float32(1.0), theta).astype(np.float32)
    k = aa / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [
            np.stack([zero, -kz, ky], axis=-1),
            np.stack([kz, zero, -kx], axis=-1),
            np.stack([-ky, kx, zero], axis=-1),
        ],
        axis=-2,
    )
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    eye = np.eye(3, dtype=np.float32)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

    ax, ay, az = aa[..., 0], aa[..., 1], aa[..., 2]
    Kaa = np.stack(
        [
            np.stack([zero, -az, ay], axis=-1),
            np.stack([az, zero, -ax], axis=-1),
            np.stack([-ay, ax, zero], axis=-1),
        ],
        axis=-2,
    )
    R_small = eye + Kaa + 0.5 * (Kaa @ Kaa)
    R = np.where(theta[..., None] < 1e-6, R_small, R).astype(np.float32)
    return R[..., :2, :].reshape(*aa.shape[:-1], 6)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Inverse of the ``axis_angle_to_rotation_6d`` row convention.

    Gram-Schmidt orthogonalisation:
        b1 = normalize(d6[..., 0:3])
        b2 = normalize(d6[..., 3:6] - (b1·d6[..., 3:6]) * b1)
        b3 = cross(b1, b2)
        R  = stack([b1, b2, b3], axis=-2)   # rows
    """
    d6 = np.asarray(d6, dtype=np.float32)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Inverse of Rodrigues. Returns axis-angle (rotation vector) of shape (..., 3).

    Numerically stable in the small-angle and near-π regimes.
    """
    R = np.asarray(R, dtype=np.float32)
    # cos(theta) = (trace(R) - 1) / 2, clamped for numerical safety
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)  # in [0, pi]

    # Off-diagonal differences encode sin(theta) * axis
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    off = np.stack([rx, ry, rz], axis=-1)
    sin_t = np.sin(theta)[..., None]

    # General case
    axis_general = off / np.clip(2.0 * sin_t, 1e-8, None)
    rotvec_general = axis_general * theta[..., None]

    # Small-angle: R ≈ I + skew(rotvec)  =>  rotvec ≈ 0.5 * off
    rotvec_small = 0.5 * off

    # Near-pi: use diagonal-based recovery
    # diag = 1 - 2*sin^2(theta/2) per axis ... here we use (R + R.T)/2 - cos_t*I
    diag = np.stack(
        [R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]], axis=-1
    )
    axis_pi = np.sqrt(np.clip((diag - cos_t[..., None]) / (1.0 - cos_t[..., None] + 1e-12), 0.0, 1.0))
    # restore signs from off-diagonals
    sign = np.sign(off)
    sign = np.where(sign == 0, 1.0, sign)
    axis_pi = axis_pi * sign
    rotvec_pi = axis_pi * theta[..., None]

    small = (theta < 1e-6)[..., None]
    near_pi = (theta > np.pi - 1e-3)[..., None]
    rotvec = np.where(small, rotvec_small, rotvec_general)
    rotvec = np.where(near_pi, rotvec_pi, rotvec)
    return rotvec.astype(np.float32)


def rotation_6d_to_axis_angle(d6: np.ndarray) -> np.ndarray:
    return rotation_matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def decode_abs_action_10d_to_7d(action_10d: np.ndarray) -> np.ndarray:
    """Convert policy output ``[xyz_abs(3), rotation_6d(6), gripper(1)]`` to the
    7D ``[xyz_abs(3), axis_angle(3), gripper(1)]`` that an OSC_POSE controller
    in ``control_delta=False`` mode expects.
    """
    action_10d = np.asarray(action_10d, dtype=np.float32)
    if action_10d.shape[-1] != 10:
        raise ValueError(
            f"decode_abs_action_10d_to_7d expects last dim 10, got {action_10d.shape[-1]}"
        )
    pos = action_10d[..., 0:3]
    rot6d = action_10d[..., 3:9]
    grip = action_10d[..., 9:10]
    axis_angle = rotation_6d_to_axis_angle(rot6d)
    return np.concatenate([pos, axis_angle, grip], axis=-1).astype(np.float32)


def state_to_abs_action(
    states: np.ndarray,
    raw_actions: np.ndarray,
    *,
    pos_slice: tuple[int, int] = (0, 3),
    rot_slice: tuple[int, int] = (3, 6),
) -> np.ndarray:
    """Convert an episode's (state, raw_action) tracks into 10D abs actions.

    For each frame ``t``:
        target_pos = state[t+1, pos_slice]
        target_rot = axis_angle_to_rotation_6d(state[t+1, rot_slice])
        gripper    = raw_action[t, -1]
    For the terminal frame ``t = T-1`` we fall back to ``state[t]`` (no-op).
    """
    states = np.asarray(states, dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if states.shape[0] != raw_actions.shape[0]:
        raise ValueError(
            f"states and raw_actions must share T; got {states.shape[0]} vs {raw_actions.shape[0]}"
        )
    next_states = np.concatenate([states[1:], states[-1:]], axis=0)
    pos_lo, pos_hi = pos_slice
    rot_lo, rot_hi = rot_slice
    target_pos = next_states[:, pos_lo:pos_hi]
    target_axis_angle = next_states[:, rot_lo:rot_hi]
    target_rot_6d = axis_angle_to_rotation_6d(target_axis_angle)
    gripper = raw_actions[:, -1:]
    return np.concatenate([target_pos, target_rot_6d, gripper], axis=-1).astype(np.float32)
