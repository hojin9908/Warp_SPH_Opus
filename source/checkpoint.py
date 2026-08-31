"""Recursive checkpoint / replay 로 다중 스텝 역전파를 한다.

Warp Tape 는 커널 런치를 기록한다. 스텝마다 상태 배열과 HashGrid 를 새로 잡아야
그래디언트가 맞는데, 그러면 메모리가 O(T) 로 자란다.
구간을 나눠 경계 상태만 저장하고 역방향에서 구간을 다시 기록하면 이것을 줄일 수 있다.

깊이 r 로 재귀하면
    메모리  ~ (r+1) * T^(1/(r+1)) * S,   정방향 계산량 ~ (r+1) * T
가 된다. r=0 이면 전 구간을 한 테이프에 올리는 기준 구현이 된다.

저장하는 상태는 (pos, vel) 뿐이다. h 는 실행 전에 정해지는 입력이라
스텝 사이에 변하지 않는다.
"""

import math

import warp as wp

from source import simulation as sim
from source.config import Config


def segment_length(cfg: Config, n: int, depth: int) -> int:
    """깊이 depth 에서 한 세그먼트의 길이.

    세그먼트 개수가 n^(1/(depth+1)) 이 되게 잡는다.
    graph_in_grad 를 켰을 때만 그래프 블록의 배수로 맞춘다. 그래야 세그먼트
    시작점과 길이가 블록에 정렬되어 1단계 전진이 그래프를 탈 수 있다.

    cfg: 설정값 묶음
    n: 이 구간의 스텝 수
    depth: 남은 checkpointing 깊이
    """
    L = math.ceil(n ** (depth / (depth + 1.0)))
    L = max(L, cfg.ckpt_min_segment)
    if cfg.use_cuda_graph and cfg.graph_in_grad:
        b = cfg.graph_block
        L = max(b, int(round(L / b)) * b)
    return min(L, n)


def backward_taped(
    cfg: Config,
    pos0: wp.array,             # [#all, 3]
    vel0: wp.array,             # [#all, 3]
    ptl_type: wp.array,         # [#all]
    n_steps: int,
    step0: int,
    seed_gpos: wp.array,        # [#all, 3]
    seed_gvel: wp.array,        # [#all, 3]
) -> tuple[wp.array, wp.array]:
    """n_steps 전부를 한 테이프에 기록하고 역전파한다 (재귀의 바닥).

    cfg: 설정값 묶음
    pos0, vel0: 구간 시작 상태 (requires_grad=True 여야 한다)
    ptl_type: 입자 종류
    n_steps: 이 구간의 스텝 수
    step0: 전역 스텝 번호의 시작값
    seed_gpos, seed_gvel: 구간 끝 상태의 adjoint

    return: (dL/dpos0, dL/dvel0)  # 새로 할당한 배열
    """
    n = ptl_type.shape[0]

    # 스텝마다 별도의 상태 배열 / 작업 버퍼 / HashGrid 를 잡는다.
    pos = [pos0] + [wp.zeros(n, dtype=wp.vec3, requires_grad=True)
                    for _ in range(n_steps)]                # [(n_steps+1)][#all, 3]
    vel = [vel0] + [wp.zeros(n, dtype=wp.vec3, requires_grad=True)
                    for _ in range(n_steps)]                # [(n_steps+1)][#all, 3]
    works = [sim.Work(n, requires_grad=True) for _ in range(n_steps)]
    grids = [sim.new_grid(cfg) for _ in range(n_steps)]     # 파이썬 리스트로 붙잡아 둔다

    # 테이프에 기록하는 구간은 CUDA graph 로 묶지 않는다. 테이프가 런치를 파이썬
    # 쪽에서 기록해야 하고, backward 가 그 기록을 되짚어 다시 런치하기 때문이다.
    tape = wp.Tape()
    for t in range(n_steps):
        sim.grid_build(cfg, grids[t], pos[t])               # tape 밖
        with tape:
            sim.sph_step(cfg, grids[t], pos[t], vel[t], works[t],
                         pos[t + 1], vel[t + 1], ptl_type, step0 + t)

    # 구간 시작의 그래디언트를 깨끗하게 만든 뒤 역전파한다.
    pos0.grad.zero_()
    vel0.grad.zero_()
    tape.backward(grads={pos[n_steps]: seed_gpos, vel[n_steps]: seed_gvel})

    g_pos = wp.clone(pos0.grad, requires_grad=False)        # [#all, 3]
    g_vel = wp.clone(vel0.grad, requires_grad=False)        # [#all, 3]
    tape.zero()
    return g_pos, g_vel


def backward_rollout(
    cfg: Config,
    pos0: wp.array,             # [#all, 3]
    vel0: wp.array,             # [#all, 3]
    ptl_type: wp.array,         # [#all]
    n_steps: int,
    step0: int,
    seed_gpos: wp.array,        # [#all, 3]
    seed_gvel: wp.array,        # [#all, 3]
    depth: int,
    roll: sim.Rollout | None = None,
) -> tuple[wp.array, wp.array]:
    """[step0, step0+n_steps) 구간을 역전파한다.

    cfg: 설정값 묶음
    pos0, vel0: 구간 시작 상태 (requires_grad=True 여야 한다)
    ptl_type: 입자 종류
    n_steps: 이 구간의 스텝 수
    step0: 전역 스텝 번호의 시작값
    seed_gpos, seed_gvel: 구간 끝 상태의 adjoint
    depth: 남은 checkpointing 깊이
    roll: 1단계 전진에 쓸 작업 공간. 재귀 전체가 하나를 돌려 쓰면
        CUDA graph 를 한 번만 캡처한다

    return: 구간 시작 상태의 adjoint
    """
    if depth <= 0 or n_steps <= cfg.ckpt_min_segment:
        return backward_taped(cfg, pos0, vel0, ptl_type, n_steps, step0,
                              seed_gpos, seed_gvel)

    n = ptl_type.shape[0]
    if roll is None:
        roll = sim.Rollout(cfg, n)
    L = segment_length(cfg, n_steps, depth)
    K = math.ceil(n_steps / L)

    # --- 1단계: 테이프 없이 전진하며 세그먼트 경계 상태만 저장한다 ---
    ck_pos = [pos0]                                          # [(K+1)][#all, 3]
    ck_vel = [vel0]                                          # [(K+1)][#all, 3]
    for seg in range(K):
        m = min(L, n_steps - seg * L)
        pos_n, vel_n, _ = sim.simulate(cfg, ck_pos[seg], ck_vel[seg], ptl_type, m,
                                       step0=step0 + seg * L, roll=roll,
                                       allow_graph=cfg.graph_in_grad)
        ck_pos.append(wp.clone(pos_n, requires_grad=True))
        ck_vel.append(wp.clone(vel_n, requires_grad=True))

    # --- 2단계: 뒤에서부터 세그먼트를 하나씩 다시 기록하며 역전파한다 ---
    g_pos, g_vel = seed_gpos, seed_gvel
    for seg in reversed(range(K)):
        m = min(L, n_steps - seg * L)
        g_pos, g_vel = backward_rollout(
            cfg, ck_pos[seg], ck_vel[seg], ptl_type, m, step0 + seg * L,
            g_pos, g_vel, depth - 1, roll,
        )
    return g_pos, g_vel


def checkpoint_report(cfg: Config, n_steps: int) -> str:
    """설정된 깊이에서 저장되는 상태 개수와 정방향 재계산 횟수를 미리 보여 준다."""
    lines: list[str] = []
    n = n_steps
    depth = cfg.ckpt_depth
    stored = 0
    while depth > 0 and n > cfg.ckpt_min_segment:
        L = segment_length(cfg, n, depth)
        K = math.ceil(n / L)
        lines.append(f"  level {cfg.ckpt_depth - depth + 1}: {K} segments x {L} steps "
                     f"-> {K + 1} checkpoints")
        stored += K + 1
        n = L
        depth -= 1
    lines.append(f"  taped segment: {n} steps -> {n + 1} states on tape")
    stored += n + 1
    lines.append(f"  총 저장 상태 ~ {stored} (전부 저장하면 {n_steps + 1}), "
                 f"정방향 계산량 ~ {cfg.ckpt_depth + 1}T")
    return "\n".join(lines)
