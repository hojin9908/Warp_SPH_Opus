"""초기 입자 배치.

내부 영역은 0 <= x <= tank_width, 0 <= y <= tank_height 이고
경계는 그 바깥에 dx 간격으로 bnd_layer 겹 쌓은 dummy particle 이다.
위쪽은 열려 있다.
"""

import numpy as np
import warp as wp

from source.config import Config
from source.read_input import BND, PTL


class particle_generation:
    """2D dam break 의 초기 입자 정보를 만든다.

    cfg: 설정값 묶음
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dx: float = cfg.dx
        self.spacing: float = cfg.dx        # 입자 간격 (경계도 같은 간격을 쓴다)

    def init_particle(self) -> np.ndarray:
        """유체 블록을 격자로 채운다.

        return: 유체 입자 위치            # [#ptl, 2]
        """
        cfg = self.cfg
        n_x = int(round(cfg.fluid_width / self.dx))
        n_y = int(round(cfg.fluid_height / self.dx))
        gx, gy = np.meshgrid(
            cfg.fluid_origin_x + np.arange(n_x) * self.dx,
            cfg.fluid_origin_y + np.arange(n_y) * self.dx,
            indexing="ij",
        )
        return np.stack([gx.ravel(), gy.ravel()], axis=1)        # [#ptl, 2]

    def dam2d_boundary(self) -> np.ndarray:
        """바닥 + 좌우 벽을 bnd_layer 겹으로 쌓는다. 위쪽은 열어 둔다.

        return: 경계 입자 위치            # [#bnd, 2]
        """
        cfg = self.cfg
        dx = self.dx
        n_x = int(round(cfg.tank_width / dx)) + 1
        n_y = int(round(cfg.tank_height / dx)) + 1

        walls: list[np.ndarray] = []
        for k in range(1, cfg.bnd_layer + 1):
            # 바닥 (모서리를 덮도록 좌우로 bnd_layer 칸 더 뺀다)
            xs = np.arange(-cfg.bnd_layer, n_x + cfg.bnd_layer) * dx
            walls.append(np.stack([xs, np.full_like(xs, -k * dx)], axis=1))
            # 좌/우 벽
            ys = np.arange(0, n_y) * dx
            walls.append(np.stack([np.full_like(ys, -k * dx), ys], axis=1))
            walls.append(np.stack([np.full_like(ys, cfg.tank_width + k * dx), ys], axis=1))
        return np.concatenate(walls, axis=0)                     # [#bnd, 2]

    def build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """유체와 경계를 한 배열로 합친다.

        유체를 앞에, 경계를 뒤에 둔다. HashGrid 는 전체 입자 위에 하나만 만든다.
        결과는 read_input() 이 파일에서 읽어 오는 것과 같은 모양이다.

        return:
            base_pos  # [#all, 3]  z 성분은 0
            base_vel  # [#all, 3]  생성할 때는 전부 0
            ptl_type  # [#all]     PTL 또는 BND
            n_ptl     # 유체 입자 수
        """
        ptl = self.init_particle()                               # [#ptl, 2]
        bnd = self.dam2d_boundary()                              # [#bnd, 2]
        xy = np.concatenate([ptl, bnd], axis=0)                  # [#all, 2]
        ptl_type = np.concatenate(
            [np.full(len(ptl), PTL), np.full(len(bnd), BND)]
        ).astype(np.int32)                                       # [#all]

        if self.cfg.jitter > 0.0:
            rng = np.random.default_rng(self.cfg.seed)
            xy = xy + rng.normal(0.0, self.cfg.jitter * self.dx, size=xy.shape)

        base_pos = np.zeros((len(xy), 3), dtype=np.float32)      # [#all, 3]
        base_pos[:, 0] = xy[:, 0]
        base_pos[:, 1] = xy[:, 1]
        base_vel = np.zeros((len(xy), 3), dtype=np.float32)      # [#all, 3]
        return base_pos, base_vel, ptl_type, len(ptl)


def to_warp(base_pos: np.ndarray, base_vel: np.ndarray,
            ptl_type: np.ndarray) -> tuple[wp.array, wp.array, wp.array]:
    """numpy 배열을 Warp 배열로 올린다.

    base_pos: 초기 위치     # [#all, 3]
    base_vel: 초기 속도     # [#all, 3]
    ptl_type: 입자 종류     # [#all]
    """
    return (wp.array(base_pos, dtype=wp.vec3),
            wp.array(base_vel, dtype=wp.vec3),
            wp.array(ptl_type, dtype=wp.int32))
