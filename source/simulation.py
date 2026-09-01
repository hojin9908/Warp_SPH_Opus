"""한 스텝과 순방향 시뮬레이션.

sph_step() 하나만 읽으면 SPH 한 스텝의 전부가 보인다.
테이프에 기록할 때는 호출부에서 `with tape:` 로 감싸기만 하면 된다.

테이프 없는 전진(simulate)은 Rollout 작업 공간을 돌려 써서 호출마다 배열과
HashGrid 를 새로 잡지 않는다.
"""

import numpy as np
import warp as wp

from source import kernel as kn
from source.config import Config


class Work:
    """한 스텝의 중간값을 담는 작업 버퍼.

    n: 전체 입자 수 (#all)
    requires_grad: 자동미분 대상으로 만들지
    """

    def __init__(self, n: int, requires_grad: bool = False) -> None:
        self.rho_raw = wp.zeros(n, dtype=float, requires_grad=requires_grad)     # [#all]
        self.rho = wp.zeros(n, dtype=float, requires_grad=requires_grad)         # [#all]
        self.pres = wp.zeros(n, dtype=float, requires_grad=requires_grad)        # [#all]
        self.acc = wp.zeros(n, dtype=wp.vec3, requires_grad=requires_grad)       # [#all, 3]


def new_grid(cfg: Config) -> wp.HashGrid:
    """HashGrid 하나. 셀 크기는 build 의 radius(=support) 가 정한다."""
    return wp.HashGrid(cfg.grid_dim, cfg.grid_dim, 1)


def grid_build(cfg: Config, grid: wp.HashGrid, pos: wp.array) -> None:
    """이웃 탐색 준비. 반드시 tape 밖에서 부른다 (미분 대상이 아니다).

    pos: 현재 위치      # [#all, 3]
    """
    grid.build(points=pos, radius=cfg.support)


def sph_step(
    cfg: Config,
    grid: wp.HashGrid,
    pos_in: wp.array,           # [#all, 3]
    vel_in: wp.array,           # [#all, 3]
    work: Work,
    pos_out: wp.array,          # [#all, 3]
    vel_out: wp.array,          # [#all, 3]
    ptl_type: wp.array,         # [#all]
    step: int,
) -> None:
    """이웃 탐색(이미 끝남) -> 밀도 -> 압력 -> 힘 -> 적분.

    cfg: 설정값 묶음
    grid: 이번 스텝의 위치로 build 가 끝난 HashGrid
    pos_in, vel_in: 스텝 시작 상태
    work: 중간값 버퍼
    pos_out, vel_out: 스텝 결과 (입력과 반드시 다른 배열이어야 한다)
    ptl_type: 입자 종류
    step: 전역 스텝 번호. Shepard filter 주기 판정에 쓴다
    """
    n = ptl_type.shape[0]
    use_shepard = cfg.shepard and cfg.shepard_step > 0 and step % cfg.shepard_step == 0

    # 1) 밀도
    wp.launch(
        kn.density_cal,
        dim=n,
        inputs=[grid.id, pos_in, cfg.h, cfg.mass, cfg.support, cfg.kernel_id],
        outputs=[work.rho_raw],
    )

    # 2) 밀도 보정 (Shepard filter). 적용하지 않는 스텝은 그대로 복사한다.
    if use_shepard:
        wp.launch(
            kn.shepard_filter,
            dim=n,
            inputs=[grid.id, pos_in, cfg.h, work.rho_raw, cfg.mass, cfg.support,
                    cfg.kernel_id],
            outputs=[work.rho],
        )
    else:
        wp.copy(work.rho, work.rho_raw)     # wp.copy 는 미분 가능하다

    # 3) 압력
    wp.launch(
        kn.pres_cal,
        dim=n,
        inputs=[work.rho, cfg.rho0, cfg.B, cfg.gamma, int(cfg.clamp_negative_pressure)],
        outputs=[work.pres],
    )

    # 4) 힘 (압력 + 점성 + 중력)
    wp.launch(
        kn.force_cal,
        dim=n,
        inputs=[grid.id, pos_in, vel_in, cfg.h, work.rho, work.pres, cfg.mass,
                cfg.support, cfg.kernel_id, cfg.mu, cfg.g],
        outputs=[work.acc],
    )

    # 5) 적분
    wp.launch(
        kn.vel_pos_step,
        dim=n,
        inputs=[pos_in, vel_in, work.acc, ptl_type, cfg.dt],
        outputs=[pos_out, vel_out],
    )


class Rollout:
    """테이프 없는 전진용 작업 공간.

    같은 Rollout 을 여러 simulate 호출에 넘기면 버퍼와 HashGrid 를 돌려 쓴다.

    cfg: 설정값 묶음
    n: 전체 입자 수 (#all)
    """

    def __init__(self, cfg: Config, n: int) -> None:
        self.pos_a = wp.zeros(n, dtype=wp.vec3)     # [#all, 3]
        self.pos_b = wp.zeros(n, dtype=wp.vec3)     # [#all, 3]
        self.vel_a = wp.zeros(n, dtype=wp.vec3)     # [#all, 3]
        self.vel_b = wp.zeros(n, dtype=wp.vec3)     # [#all, 3]
        self.work = Work(n)
        self.grid = new_grid(cfg)


def simulate(
    cfg: Config,
    pos_start: wp.array,        # [#all, 3]
    vel_start: wp.array,        # [#all, 3]
    ptl_type: wp.array,         # [#all]
    n_steps: int,
    step0: int = 0,
    snapshot_step: int = 0,
    roll: Rollout | None = None,
) -> tuple[wp.array, wp.array, list[tuple[np.ndarray, np.ndarray]]]:
    """테이프 없이 n_steps 전진한다.

    cfg: 설정값 묶음
    pos_start, vel_start: 시작 상태
    ptl_type: 입자 종류
    n_steps: 전진할 스텝 수
    step0: 전역 스텝 번호의 시작값 (Shepard 주기 판정용)
    snapshot_step: 몇 스텝마다 상태를 기록할지. 0 이면 기록하지 않는다
    roll: 작업 공간. 넘기면 버퍼와 HashGrid 를 돌려 쓴다

    return:
        pos_final, vel_final    # [#all, 3]  항상 새 배열로 복사해서 준다
                                #   (roll 을 공유하면 내부 버퍼가 다음 호출에 덮어써진다)
        snapshots               # [(pos numpy, vel numpy), ...]
    """
    n = ptl_type.shape[0]
    if roll is None:
        roll = Rollout(cfg, n)
    roll.pos_a.assign(pos_start)
    roll.vel_a.assign(vel_start)

    snaps: list[tuple[np.ndarray, np.ndarray]] = []
    pos_a, pos_b = roll.pos_a, roll.pos_b
    vel_a, vel_b = roll.vel_a, roll.vel_b
    for s in range(n_steps):
        if snapshot_step > 0 and s % snapshot_step == 0:
            snaps.append((pos_a.numpy().copy(), vel_a.numpy().copy()))
        grid_build(cfg, roll.grid, pos_a)
        sph_step(cfg, roll.grid, pos_a, vel_a, roll.work, pos_b, vel_b,
                 ptl_type, step0 + s)
        pos_a, pos_b = pos_b, pos_a
        vel_a, vel_b = vel_b, vel_a

    if snapshot_step > 0:
        snaps.append((pos_a.numpy().copy(), vel_a.numpy().copy()))
    return (wp.clone(pos_a, requires_grad=False),
            wp.clone(vel_a, requires_grad=False),
            snaps)
