"""그래디언트 검증.

Warp 는 틀린 그래디언트를 에러 없이 돌려주는 경우가 있으므로 (동적 루프의 비누적
연산, HashGrid 재사용 등) 반드시 실제 dt 로 유한차분과 대조한다.

세 가지를 본다.
  1) 궤적이 얼마나 빨리 발산하는가 (미분이 쓸모 있는 구간을 정한다)
  2) recursive checkpointing 과 full-tape 이 같은 값을 주는가
  3) 자동미분과 중심 유한차분이 얼마나 긴 구간까지 일치하는가
"""

import copy

import numpy as np
import warp as wp

from source import optimize as opt
from source import simulation as sim
from source.config import Config
from source.gen_ptl import PTL, particle_generation, to_warp


def grad_of_offset(
    cfg: Config,
    base_pos: wp.array,         # [#all, 3]
    ptl_type: wp.array,         # [#all]
    pos_target: wp.array,       # [#all, 3]
    n_ptl: int,
    n_steps: int,
    ox: float,
    oy: float,
    depth: int,
) -> tuple[float, np.ndarray]:
    """주어진 checkpointing 깊이로 loss 와 dL/d(offset) 을 구한다."""
    c = copy.copy(cfg)
    c.ckpt_depth = depth
    offset = opt.offset_array(ox, oy)
    loss, _ = opt.loss_and_grad(c, offset, base_pos, ptl_type, pos_target, n_ptl, n_steps)
    return loss, offset.grad.numpy()[0][:2].copy()


def loss_only(
    cfg: Config,
    base_pos: wp.array,         # [#all, 3]
    ptl_type: wp.array,         # [#all]
    pos_target: wp.array,       # [#all, 3]
    n_ptl: int,
    n_steps: int,
    ox: float,
    oy: float,
) -> float:
    """그래디언트 없이 loss 만 계산한다 (유한차분용)."""
    offset = opt.offset_array(ox, oy)
    s0 = opt.ptl_place(cfg, base_pos, offset, ptl_type)
    s_final, _ = sim.simulate(cfg, s0, ptl_type, n_steps)
    return opt.loss_value(s_final, pos_target, ptl_type, 1.0 / n_ptl)


def finite_difference(
    cfg: Config,
    base_pos: wp.array,         # [#all, 3]
    ptl_type: wp.array,         # [#all]
    pos_target: wp.array,       # [#all, 3]
    n_ptl: int,
    n_steps: int,
    ox: float,
    oy: float,
    eps: float,
) -> np.ndarray:
    """중심 유한차분으로 dL/d(offset) 을 구한다.

    return: [2]
    """
    g = np.zeros(2)
    for d in range(2):
        p = [ox, oy]
        p[d] += eps
        lp = loss_only(cfg, base_pos, ptl_type, pos_target, n_ptl, n_steps, p[0], p[1])
        p = [ox, oy]
        p[d] -= eps
        lm = loss_only(cfg, base_pos, ptl_type, pos_target, n_ptl, n_steps, p[0], p[1])
        g[d] = (lp - lm) / (2.0 * eps)
    return g


def run_check(
    cfg: Config,
    horizons: tuple[int, ...] = (100, 200, 400, 800),
    eps: float = 5.0e-4,
    probe_offset: tuple[float, float] = (0.02, 0.01),
    amp_limit: float = 2.0,
    fd_limit: float = 0.05,
) -> bool:
    """검증을 돌리고 통과 여부를 돌려준다.

    cfg: 설정값 묶음
    horizons: 검사할 시뮬레이션 길이들
    eps: 유한차분 간격
    probe_offset: 그래디언트를 평가할 지점
    amp_limit: 이 증폭률 이하 구간에서만 checkpointing 일치를 판정한다
    fd_limit: 자동미분과 유한차분이 이 상대오차 안이면 "쓸 만한 구간"으로 본다
    """
    gen = particle_generation(cfg)
    base_np, ptl_type_np, n_ptl = gen.build()
    base_pos, ptl_type = to_warp(base_np, ptl_type_np)
    n = len(ptl_type_np)
    is_ptl = ptl_type_np == PTL
    ox, oy = probe_offset
    print(f"입자 {n} 개 (유체 {n_ptl}), dt = {cfg.dt:.3e}, 평가점 offset = ({ox}, {oy})")
    print(f"kernel={cfg.kernel_type}  h={cfg.h:.5f} (h/dx={cfg.h / cfg.dx:.2f})")

    def final_pos(T: int, a: float, b: float) -> np.ndarray:
        """T 스텝 뒤의 위치를 numpy 로 돌려준다."""
        offset = opt.offset_array(a, b)
        s0 = opt.ptl_place(cfg, base_pos, offset, ptl_type)
        s_final, _ = sim.simulate(cfg, s0, ptl_type, T)
        return s_final.pos

    # ---------- 1) 궤적 발산: 어디까지가 미분이 쓸모 있는 구간인지 먼저 잰다 ----------
    print("\n[1] 궤적 발산: offset 을 1e-6 만큼 바꿨을 때 최종 위치 차이의 증폭률")
    print(f"{'T':>6} {'mean|dx|':>12} {'max|dx|':>12} {'증폭':>12}")
    amp: dict[int, float] = {}
    for T in horizons:
        a = final_pos(T, ox, oy).numpy()
        b = final_pos(T, ox + 1e-6, oy).numpy()
        d = np.linalg.norm((a - b)[is_ptl, :2], axis=1)
        amp[T] = float(d.mean() / 1e-6)
        print(f"{T:6d} {d.mean():12.3e} {d.max():12.3e} {amp[T]:12.3e}")

    # ---------- 2) checkpointing 일치 + AD vs FD ----------
    print(f"\n[2] full-tape(depth=0) vs recursive checkpointing(depth={cfg.ckpt_depth}) "
          f"vs 중심 유한차분(eps={eps})")
    print(f"{'T':>6} {'loss':>13} | {'AD full':>21} | {'AD ckpt':>21} | "
          f"{'FD':>21} | {'ckpt-full':>9} {'AD-FD':>9} | 판정")
    ok = True
    usable = 0
    for T in horizons:
        pos_target = final_pos(T, cfg.true_offset_x, cfg.true_offset_y)
        l_full, g_full = grad_of_offset(cfg, base_pos, ptl_type, pos_target, n_ptl,
                                        T, ox, oy, 0)
        _, g_ck = grad_of_offset(cfg, base_pos, ptl_type, pos_target, n_ptl,
                                 T, ox, oy, cfg.ckpt_depth)
        g_fd = finite_difference(cfg, base_pos, ptl_type, pos_target, n_ptl,
                                 T, ox, oy, eps)

        scale = max(float(np.abs(g_full).max()), 1e-12)
        e_ck = float(np.abs(g_ck - g_full).max()) / scale
        e_fd = float(np.abs(g_full - g_fd).max()) / scale

        # 증폭률이 큰 구간은 부동소수점 차이가 지수적으로 커지므로 판정에서 뺀다.
        if amp[T] <= amp_limit:
            verdict = "PASS" if e_ck < 1e-4 else "FAIL"
            if e_ck >= 1e-4:
                ok = False
            if e_fd < fd_limit:
                usable = max(usable, T)
        else:
            verdict = "카오스 구간"
        print(f"{T:6d} {l_full:13.6e} | ({g_full[0]:+9.5f},{g_full[1]:+9.5f}) | "
              f"({g_ck[0]:+9.5f},{g_ck[1]:+9.5f}) | ({g_fd[0]:+9.5f},{g_fd[1]:+9.5f}) | "
              f"{e_ck:9.2e} {e_fd:9.2e} | {verdict}")

    print(f"\n  증폭률 <= {amp_limit} 인 구간에서 checkpointing 이 full-tape 과 "
          f"반올림 수준으로 일치: {'PASS' if ok else 'FAIL'}")
    print(f"  자동미분이 유한차분과 {fd_limit * 100:.0f}% 안에서 맞는 최대 구간: T = {usable}")
    print("\n  증폭률이 커지는 구간부터는 dam break 자체가 카오스라서, 국소 미분이")
    print("  여전히 정확하더라도 유한차분(유한한 eps)과 갈라지고 하강 방향으로도 쓸모가 없다.")
    return ok
