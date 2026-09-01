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


def new_state(n: int, requires_grad: bool = False) -> kn.State:
    """빈 입자 상태 하나. n 은 전체 입자 수 (#all)."""
    s = kn.State()
    s.pos = wp.zeros(n, dtype=wp.vec3, requires_grad=requires_grad)      # [#all, 3]
    s.vel = wp.zeros(n, dtype=wp.vec3, requires_grad=requires_grad)      # [#all, 3]
    return s


def clone_state(src: kn.State, requires_grad: bool = False) -> kn.State:
    """상태를 새 배열로 복사한다. checkpoint 를 붙잡아 둘 때 쓴다."""
    s = kn.State()
    s.pos = wp.clone(src.pos, requires_grad=requires_grad)
    s.vel = wp.clone(src.vel, requires_grad=requires_grad)
    return s


def assign_state(dst: kn.State, src: kn.State) -> None:
    """dst 배열에 src 값을 덮어쓴다 (배열 객체는 그대로 둔다)."""
    dst.pos.assign(src.pos)
    dst.vel.assign(src.vel)


def new_field(n: int, requires_grad: bool = False) -> kn.Field:
    """한 스텝의 중간값 버퍼. n 은 전체 입자 수 (#all)."""
    f = kn.Field()
    f.rho_raw = wp.zeros(n, dtype=float, requires_grad=requires_grad)    # [#all]
    f.rho = wp.zeros(n, dtype=float, requires_grad=requires_grad)        # [#all]
    f.pres = wp.zeros(n, dtype=float, requires_grad=requires_grad)       # [#all]
    f.acc = wp.zeros(n, dtype=wp.vec3, requires_grad=requires_grad)      # [#all, 3]
    return f


def new_grid(cfg: Config) -> wp.HashGrid:
    """HashGrid 하나. 셀 크기는 build 의 radius(=support) 가 정한다."""
    return wp.HashGrid(cfg.grid_dim, cfg.grid_dim, 1)


def grid_build(cfg: Config, grid: wp.HashGrid, s: kn.State) -> None:
    """이웃 탐색 준비. 반드시 tape 밖에서 부른다 (미분 대상이 아니다)."""
    grid.build(points=s.pos, radius=cfg.support)


def sph_step(
    cfg: Config,
    grid: wp.HashGrid,
    s_in: kn.State,
    fld: kn.Field,
    s_out: kn.State,
    ptl_type: wp.array,         # [#all]
    step: int,
) -> None:
    """이웃 탐색(이미 끝남) -> 밀도 -> 압력 -> 힘 -> 적분.

    cfg: 설정값 묶음
    grid: 이번 스텝의 위치로 build 가 끝난 HashGrid
    s_in: 스텝 시작 상태
    fld: 중간값 버퍼
    s_out: 스텝 결과 (s_in 과 반드시 다른 배열이어야 한다)
    ptl_type: 입자 종류
    step: 전역 스텝 번호. Shepard filter 주기 판정에 쓴다
    """
    n = ptl_type.shape[0]
    use_shepard = cfg.shepard and cfg.shepard_step > 0 and step % cfg.shepard_step == 0

    # 1) 밀도
    wp.launch(
        kn.density_cal,
        dim=n,
        inputs=[grid.id, s_in, fld, cfg.h, cfg.mass, cfg.support, cfg.kernel_id],
    )

    # 2) 밀도 보정 (Shepard filter). 적용하지 않는 스텝은 그대로 복사한다.
    if use_shepard:
        wp.launch(
            kn.shepard_filter,
            dim=n,
            inputs=[grid.id, s_in, fld, cfg.h, cfg.mass, cfg.support, cfg.kernel_id],
        )
    else:
        wp.copy(fld.rho, fld.rho_raw)       # wp.copy 는 미분 가능하다

    # 3) 압력
    wp.launch(
        kn.pres_cal,
        dim=n,
        inputs=[fld, cfg.rho0, cfg.B, cfg.gamma, int(cfg.clamp_negative_pressure)],
    )

    # 4) 힘 (압력 + 점성 + 중력)
    wp.launch(
        kn.force_cal,
        dim=n,
        inputs=[grid.id, s_in, fld, cfg.h, cfg.mass, cfg.support, cfg.kernel_id,
                cfg.mu, cfg.g],
    )

    # 5) 적분
    wp.launch(
        kn.vel_pos_step,
        dim=n,
        inputs=[s_in, fld, ptl_type, cfg.dt],
        outputs=[s_out],
    )


class Rollout:
    """테이프 없는 전진용 작업 공간.

    같은 Rollout 을 여러 simulate 호출에 넘기면 버퍼와 HashGrid 를 돌려 쓴다.

    cfg: 설정값 묶음
    n: 전체 입자 수 (#all)
    """

    def __init__(self, cfg: Config, n: int) -> None:
        self.s_a = new_state(n)
        self.s_b = new_state(n)
        self.fld = new_field(n)
        self.grid = new_grid(cfg)


def simulate(
    cfg: Config,
    s_start: kn.State,
    ptl_type: wp.array,         # [#all]
    n_steps: int,
    step0: int = 0,
    snapshot_step: int = 0,
    roll: Rollout | None = None,
) -> tuple[kn.State, list[tuple[np.ndarray, np.ndarray]]]:
    """테이프 없이 n_steps 전진한다.

    cfg: 설정값 묶음
    s_start: 시작 상태
    ptl_type: 입자 종류
    n_steps: 전진할 스텝 수
    step0: 전역 스텝 번호의 시작값 (Shepard 주기 판정용)
    snapshot_step: 몇 스텝마다 상태를 기록할지. 0 이면 기록하지 않는다
    roll: 작업 공간. 넘기면 버퍼와 HashGrid 를 돌려 쓴다

    return:
        s_final     항상 새 배열로 복사해서 준다
                    (roll 을 공유하면 내부 버퍼가 다음 호출에 덮어써진다)
        snapshots   [(pos numpy, vel numpy), ...]
    """
    n = ptl_type.shape[0]
    if roll is None:
        roll = Rollout(cfg, n)
    assign_state(roll.s_a, s_start)

    snaps: list[tuple[np.ndarray, np.ndarray]] = []
    s_a, s_b = roll.s_a, roll.s_b
    for t in range(n_steps):
        if snapshot_step > 0 and t % snapshot_step == 0:
            snaps.append((s_a.pos.numpy().copy(), s_a.vel.numpy().copy()))
        grid_build(cfg, roll.grid, s_a)
        sph_step(cfg, roll.grid, s_a, roll.fld, s_b, ptl_type, step0 + t)
        s_a, s_b = s_b, s_a

    if snapshot_step > 0:
        snaps.append((s_a.pos.numpy().copy(), s_a.vel.numpy().copy()))
    return clone_state(s_a), snaps
