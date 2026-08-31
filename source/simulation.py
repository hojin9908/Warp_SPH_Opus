"""한 스텝과 순방향 시뮬레이션.

sph_step() 하나만 읽으면 SPH 한 스텝의 전부가 보인다.
테이프에 기록할 때는 호출부에서 `with tape:` 로 감싸기만 하면 된다.

테이프 없는 전진(simulate)은 Rollout 작업 공간을 쓰고, 조건이 맞으면
스텝 여러 개를 CUDA graph 하나로 묶어 커널 런치 오버헤드를 없앤다.
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


# 캡처해 둔 CUDA graph 는 다른 HashGrid 를 만들어 쓰면 무효가 된다.
# (실측: 캡처 뒤 새 HashGrid 를 build 하고 eager 스텝을 돌리면, 그 다음 graph
#  실행 결과가 완전히 깨진다 — 유체가 수조 밖으로 날아간다. Warp 내부에서
#  HashGrid 들이 정렬 스크래치를 공유하는 것으로 보인다.)
# 그래서 grid 를 만들거나 build 할 때마다 세대 번호를 올리고, 그래프를 쏘기 전에
# 캡처 당시의 세대와 같은지 확인한다. 다르면 다시 캡처한다.
_grid_epoch: int = 0


def new_grid(cfg: Config) -> wp.HashGrid:
    """HashGrid 하나. 셀 크기는 build 의 radius(=support) 가 정한다."""
    global _grid_epoch
    _grid_epoch += 1
    return wp.HashGrid(cfg.grid_dim, cfg.grid_dim, 1)


def grid_build(cfg: Config, grid: wp.HashGrid, pos: wp.array) -> None:
    """이웃 탐색 준비. 반드시 tape 밖에서 부른다 (미분 대상이 아니다).

    pos: 현재 위치      # [#all, 3]
    """
    global _grid_epoch
    _grid_epoch += 1
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

    버퍼가 고정되어 있어야 CUDA graph 를 캡처해 두고 재사용할 수 있다.
    같은 Rollout 을 여러 simulate 호출에 넘기면 그래프를 한 번만 캡처한다.

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
        self.graph: wp.Graph | None = None
        self.graph_epoch: int = -1      # 캡처 당시의 _grid_epoch


def capture_block(cfg: Config, roll: Rollout, ptl_type: wp.array, block: int) -> None:
    """block 개 스텝을 CUDA graph 하나로 캡처한다.

    block 이 짝수라 캡처가 끝나면 ping-pong 이 제자리로 돌아오고,
    block 이 Shepard 주기의 배수라 그래프 안의 분기 패턴이 매 블록 같다.
    """
    # 캡처 전에 한 번은 실제로 실행해 둔다. HashGrid 가 첫 build 에서 내부 버퍼를
    # 지연 할당하는데, 그 할당이 캡처 중에 일어나면 안 된다.
    grid_build(cfg, roll.grid, roll.pos_a)
    sph_step(cfg, roll.grid, roll.pos_a, roll.vel_a, roll.work,
             roll.pos_b, roll.vel_b, ptl_type, 0)
    wp.synchronize_device()

    pos_a, pos_b = roll.pos_a, roll.pos_b
    vel_a, vel_b = roll.vel_a, roll.vel_b
    with wp.ScopedCapture() as capture:
        for s in range(block):
            grid_build(cfg, roll.grid, pos_a)
            sph_step(cfg, roll.grid, pos_a, vel_a, roll.work, pos_b, vel_b, ptl_type, s)
            pos_a, pos_b = pos_b, pos_a
            vel_a, vel_b = vel_b, vel_a
    roll.graph = capture.graph
    roll.graph_epoch = _grid_epoch


def can_use_graph(cfg: Config, n_steps: int, step0: int, snapshot_step: int,
                  allow_graph: bool) -> bool:
    """그래프를 쓰려면 구간이 블록 길이에 딱 맞아떨어져야 한다."""
    if not allow_graph or not cfg.use_cuda_graph or not wp.get_device(cfg.device).is_cuda:
        return False
    b = cfg.graph_block
    return (step0 % b == 0 and n_steps % b == 0
            and (snapshot_step == 0 or snapshot_step % b == 0))


def simulate(
    cfg: Config,
    pos_start: wp.array,        # [#all, 3]
    vel_start: wp.array,        # [#all, 3]
    ptl_type: wp.array,         # [#all]
    n_steps: int,
    step0: int = 0,
    snapshot_step: int = 0,
    roll: Rollout | None = None,
    allow_graph: bool = True,
) -> tuple[wp.array, wp.array, list[tuple[np.ndarray, np.ndarray]]]:
    """테이프 없이 n_steps 전진한다.

    cfg: 설정값 묶음
    pos_start, vel_start: 시작 상태
    ptl_type: 입자 종류
    n_steps: 전진할 스텝 수
    step0: 전역 스텝 번호의 시작값 (Shepard 주기 판정용)
    snapshot_step: 몇 스텝마다 상태를 기록할지. 0 이면 기록하지 않는다
    roll: 작업 공간. 넘기면 CUDA graph 를 재사용한다
    allow_graph: False 면 CUDA graph 를 쓰지 않는다. 그래디언트를 구하는 경로가
        이것을 쓴다 — 이유는 config.py 의 graph_in_grad 설명 참고

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
    if can_use_graph(cfg, n_steps, step0, snapshot_step, allow_graph):
        # 캡처 이후 다른 HashGrid 가 만들어졌다면 그래프가 무효다. 다시 캡처한다.
        if roll.graph is None or roll.graph_epoch != _grid_epoch:
            capture_block(cfg, roll, ptl_type, cfg.graph_block)
        for s in range(0, n_steps, cfg.graph_block):
            if snapshot_step > 0 and s % snapshot_step == 0:
                snaps.append((roll.pos_a.numpy().copy(), roll.vel_a.numpy().copy()))
            wp.capture_launch(roll.graph)
        pos_a, vel_a = roll.pos_a, roll.vel_a
    else:
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
