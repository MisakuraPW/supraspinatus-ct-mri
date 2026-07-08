from __future__ import annotations

from collections import deque

import numpy as np


def threshold_bone(volume: np.ndarray, threshold_hu: float = 180.0) -> np.ndarray:
    return np.asarray(volume > threshold_hu, dtype=bool)


def largest_components(mask: np.ndarray, min_voxels: int = 5000, max_components: int = 3) -> np.ndarray:
    """Keep largest 6-connected components without requiring scipy/skimage."""

    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, list[tuple[int, int, int]]]] = []
    sx, sy, sz = mask.shape
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    seeds = np.argwhere(mask)
    for seed in seeds:
        x, y, z = map(int, seed)
        if visited[x, y, z]:
            continue
        q: deque[tuple[int, int, int]] = deque([(x, y, z)])
        visited[x, y, z] = True
        voxels: list[tuple[int, int, int]] = []
        while q:
            vx, vy, vz = q.popleft()
            voxels.append((vx, vy, vz))
            for dx, dy, dz in neighbors:
                nx, ny, nz = vx + dx, vy + dy, vz + dz
                if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz and mask[nx, ny, nz] and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    q.append((nx, ny, nz))
        if len(voxels) >= min_voxels:
            components.append((len(voxels), voxels))

    components.sort(key=lambda item: item[0], reverse=True)
    out = np.zeros(mask.shape, dtype=bool)
    for _, voxels in components[:max_components]:
        coords = np.asarray(voxels, dtype=np.int32)
        out[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return out

