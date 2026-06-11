"""ContactDB object: MuJoCo spec, sampled pointcloud, surface regions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg

from mjlab_hand.utils import update_assets

# Repo root: .../mjlab_hand (src/mjlab_hand/object/object.py -> parents[3])
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTACTDB_ROOT = _PROJECT_ROOT / "asset" / "contactdb"


@dataclass(frozen=True)
class RegionClusterResult:
    """FPS + k-means style clustering on canonical points."""

    point_to_region: torch.Tensor  # (M,) long, values in [0, K-1]
    region_centers: torch.Tensor  # (K, 3) float32 in canonical (object-local) frame


def _contactdb_root() -> Path:
    return CONTACTDB_ROOT


def _fps_points(
    points: torch.Tensor, k: int, start_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Farthest point sampling. points: (N, 3), returns (indices (k,), centers (k, 3))."""
    n = points.shape[0]
    k = min(k, n)
    device, dtype = points.device, points.dtype
    dists = torch.full((n,), float("inf"), device=device, dtype=dtype)
    idx = torch.empty((k,), dtype=torch.long, device=device)
    farthest = int(start_idx) % n
    for i in range(k):
        idx[i] = farthest
        center = points[farthest : farthest + 1]
        dist2 = ((points - center) ** 2).sum(dim=-1)
        dists = torch.minimum(dists, dist2)
        farthest = int(torch.argmax(dists).item())
    return idx, points[idx]


def cluster_points_fps_kmeans(
    points: torch.Tensor,
    normals: torch.Tensor | None,
    num_regions: int,
    max_iters: int = 10,
    use_normal_in_clustering: bool = True,
    normal_weight: float = 0.5,
) -> RegionClusterResult:
    """Cluster canonical points into `num_regions` regions (TOUCHMANIP-style)."""
    m, _ = points.shape
    k = min(int(num_regions), m)
    pts = points.to(dtype=torch.float32)
    nrms = normals.to(dtype=torch.float32) if normals is not None else None

    fps_idx, centers = _fps_points(pts, k, start_idx=0)
    center_normals = None
    if nrms is not None and use_normal_in_clustering:
        center_normals = nrms[fps_idx].clone()

    for _ in range(max_iters):
        if nrms is not None and use_normal_in_clustering and center_normals is not None:
            pos_dist = torch.cdist(pts, centers)
            normal_sim = torch.mm(nrms, center_normals.T)
            normal_dist = 1.0 - normal_sim.clamp(-1.0, 1.0)
            combined = (1.0 - normal_weight) * pos_dist + normal_weight * normal_dist
            labels = combined.argmin(dim=1)
        else:
            labels = torch.cdist(pts, centers).argmin(dim=1)

        new_centers = torch.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centers[i] = pts[mask].mean(dim=0)
            else:
                new_centers[i] = centers[i]
        centers = new_centers

        if nrms is not None and use_normal_in_clustering:
            new_cn = torch.zeros(k, 3, device=pts.device, dtype=pts.dtype)
            for i in range(k):
                mask = labels == i
                if mask.any():
                    avg = nrms[mask].mean(dim=0)
                    new_cn[i] = avg / avg.norm().clamp_min(1e-8)
                else:
                    new_cn[i] = (
                        center_normals[i]
                        if center_normals is not None
                        else torch.zeros(3, device=pts.device)
                    )
            center_normals = new_cn

    return RegionClusterResult(point_to_region=labels.to(torch.long), region_centers=centers)


class ContactDBObject:
    """Load ContactDB convex MuJoCo asset, sample STL pointcloud, build surface regions."""

    def __init__(
        self,
        object_name: str,
        *,
        num_points: int = 512,
        num_regions: int = 64,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        asset_root: Path | None = None,
    ) -> None:
        self.object_name = object_name
        self.num_points = int(num_points)
        self.num_regions = int(num_regions)
        self.scale = (float(scale[0]), float(scale[1]), float(scale[2]))
        root = asset_root if asset_root is not None else _contactdb_root()
        self.asset_dir = root / object_name
        self.xml_path = self.asset_dir / f"{object_name}_convex.xml"
        self.stl_path = self.asset_dir / f"{object_name}.stl"

        if not self.xml_path.is_file():
            raise FileNotFoundError(f"Missing convex XML: {self.xml_path}")
        if not self.stl_path.is_file():
            raise FileNotFoundError(f"Missing STL: {self.stl_path}")

        pts_np, nrm_np = self._sample_mesh_stl(self.stl_path, self.num_points, self.scale)
        self.canonical_pointcloud = torch.from_numpy(pts_np).to(dtype=torch.float32)
        self.canonical_normals = torch.from_numpy(nrm_np).to(dtype=torch.float32)

        cluster = cluster_points_fps_kmeans(
            self.canonical_pointcloud,
            self.canonical_normals,
            self.num_regions,
        )
        self.point_to_region = cluster.point_to_region
        self.region_centers = cluster.region_centers
        self.num_regions_effective = int(self.region_centers.shape[0])

    @staticmethod
    def _sample_mesh_stl(
        stl_path: Path,
        count: int,
        scale: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        import trimesh

        mesh = trimesh.load(str(stl_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.apply_scale(scale)
        pts, face_idx = mesh.sample(count, return_index=True)
        normals = mesh.face_normals[face_idx].astype(np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        return pts, normals

    def build_mj_spec(self) -> mujoco.MjSpec:
        """MuJoCo spec for the free-floating object (convex hulls)."""
        spec = mujoco.MjSpec.from_file(str(self.xml_path))
        assets: dict[str, bytes] = {}
        update_assets(assets, self.asset_dir, spec.meshdir)
        spec.assets = assets

        sx, sy, sz = self.scale
        for mesh in spec.meshes:
            if mesh.name == self.object_name or mesh.name.startswith(
                f"{self.object_name}_collision_hull_"
            ):
                mesh.scale = (sx, sy, sz)

        return spec

    def spec_fn(self) -> mujoco.MjSpec:
        return self.build_mj_spec()

    def get_entity_cfg(self) -> EntityCfg:
        return EntityCfg(spec_fn=self.spec_fn)

    def nearest_region_ids(
        self,
        contact_pos_world: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Map world-space contact positions to region ids.

        Args:
            contact_pos_world: (N, 3) or (N, L, 3)
            object_pos_w: (N, 3)
            object_quat_w: (N, 4) w,x,y,z

        Returns:
            Same leading shape as contact_pos_world[..., :3], long tensor of region indices.
        """
        from mjlab.utils.lab_api.math import quat_apply_inverse

        if contact_pos_world.dim() == 2:
            p_w = contact_pos_world
            expanded = False
        else:
            n, l, _ = contact_pos_world.shape
            p_w = contact_pos_world.reshape(n * l, 3)
            pos_exp = object_pos_w.unsqueeze(1).expand(n, l, 3).reshape(n * l, 3)
            quat_exp = object_quat_w.unsqueeze(1).expand(n, l, 4).reshape(n * l, 4)
            object_pos_w = pos_exp
            object_quat_w = quat_exp
            expanded = True

        rel = p_w - object_pos_w
        p_local = quat_apply_inverse(object_quat_w, rel)

        pc = self.canonical_pointcloud.to(device=p_local.device, dtype=p_local.dtype)
        regions = self.point_to_region.to(device=p_local.device)

        # (B, M)
        dist2 = torch.cdist(p_local.unsqueeze(0), pc.unsqueeze(0)).squeeze(0)
        nn = dist2.argmin(dim=-1)
        out = regions[nn]

        if not expanded:
            return out
        return out.view(n, l)


def make_contactdb_spec_fn(
    object_name: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_convex: bool = True,
) -> Callable[[], mujoco.MjSpec]:
    """Build a spec_fn for EntityCfg that loads the full `{object_name}.xml`.

    When *use_convex* is True (default), the convex-decomposition XML
    ``{object_name}_convex.xml`` is preferred if it exists, falling back to
    the single-mesh ``{object_name}.xml`` otherwise.
    """

    def _spec() -> mujoco.MjSpec:
        asset_dir = _contactdb_root() / object_name
        convex_xml_path = asset_dir / f"{object_name}_convex.xml"
        single_xml_path = asset_dir / f"{object_name}.xml"

        if use_convex and convex_xml_path.is_file():
            xml_path = convex_xml_path
        elif single_xml_path.is_file():
            xml_path = single_xml_path
        else:
            raise FileNotFoundError(
                f"Missing XML for '{object_name}': {single_xml_path}\n"
                f"Available objects: {sorted([p.name for p in _contactdb_root().iterdir() if p.is_dir() and (p / f'{p.name}.xml').is_file()])}"
            )
        spec = mujoco.MjSpec.from_file(str(xml_path))
        assets: dict[str, bytes] = {}
        update_assets(assets, asset_dir, spec.meshdir)
        spec.assets = assets

        sx, sy, sz = (float(scale[0]), float(scale[1]), float(scale[2]))
        scaled = False
        for mesh in spec.meshes:
            if mesh.name == object_name or mesh.name.startswith(f"{object_name}_collision_hull_"):
                mesh.scale = (sx, sy, sz)
                scaled = True
        if not scaled:
            raise ValueError(f"No meshes matching '{object_name}' found in {xml_path}")
        return spec

    return _spec


__all__ = [
    "CONTACTDB_ROOT",
    "ContactDBObject",
    "RegionClusterResult",
    "cluster_points_fps_kmeans",
    "make_contactdb_spec_fn",
]
