"""초기 유체 블록 위치 복원.

목표 상태 pos_target 은 참값 offset 으로 돌린 시뮬레이션의 최종 상태다.
최적화는  초기화 -> forward simulation -> loss -> backward -> 위치 갱신  을 반복한다.
"""

from typing import Any

import numpy as np
import warp as wp
from warp.optim import Adam

from source import checkpoint as ckpt
from source import kernel as kn
from source import simulation as sim
from source.config import Config


def offset_array(x: float, y: float) -> wp.array:
    """최적화 대상인 초기 이동량을 길이 1 짜리 wp.vec3 배열로 만든다.

    x, y: 초기 이동량
    return: offset      # [1, 3]
    """
    return wp.array([wp.vec3(float(x), float(y), 0.0)], dtype=wp.vec3, requires_grad=True)


def clamp_offset(cfg: Config, offset: wp.array) -> None:
    """유체 블록이 수조 안에 남도록 offset 을 가둔다.

    offset: 초기 이동량     # [1, 3]
    """
    o = offset.numpy()
    o[0][0] = min(max(o[0][0], cfg.offset_lo[0]), cfg.offset_hi[0])
    o[0][1] = min(max(o[0][1], cfg.offset_lo[1]), cfg.offset_hi[1])
    offset.assign(o)


def ptl_place(
    cfg: Config,
    base_pos: wp.array,         # [#all, 3]
    offset: wp.array,           # [1, 3]
    ptl_type: wp.array,         # [#all]
    tape: wp.Tape | None = None,
) -> wp.array:
    """offset 으로부터 초기 위치를 만든다. tape 를 주면 기록한다.

    return: pos0        # [#all, 3]
    """
    n = ptl_type.shape[0]
    pos0 = wp.zeros(n, dtype=wp.vec3, requires_grad=True)
    if tape is None:
        wp.launch(kn.place_ptl, dim=n, inputs=[base_pos, offset, ptl_type], outputs=[pos0])
    else:
        with tape:
            wp.launch(kn.place_ptl, dim=n, inputs=[base_pos, offset, ptl_type],
                      outputs=[pos0])
    return pos0


def loss_value(
    pos: wp.array,              # [#all, 3]
    pos_target: wp.array,       # [#all, 3]
    ptl_type: wp.array,         # [#all]
    inv_n: float,
) -> float:
    """유체 입자 최종 위치의 평균제곱오차를 스칼라로 돌려준다."""
    n = ptl_type.shape[0]
    loss = wp.zeros(1, dtype=float)                     # [1]
    wp.launch(kn.loss_cal, dim=n, inputs=[pos, pos_target, ptl_type, inv_n],
              outputs=[loss])
    return float(loss.numpy()[0])


def loss_and_grad(
    cfg: Config,
    offset: wp.array,           # [1, 3]
    base_pos: wp.array,         # [#all, 3]
    ptl_type: wp.array,         # [#all]
    pos_target: wp.array,       # [#all, 3]
    n_ptl: int,
    n_steps: int,
    roll: sim.Rollout | None = None,
) -> tuple[float, wp.array]:
    """loss 와 dL/d(offset) 을 계산한다.

    순서
      1) offset -> pos0            (테이프에 기록)
      2) pos0 -> pos_T             (테이프 없이 전진, loss 계산용)
      3) dL/dpos_T                 (해석적으로 씨앗을 만든다)
      4) dL/dpos_T -> dL/dpos0     (recursive checkpoint / replay)
      5) dL/dpos0  -> dL/doffset   (1) 의 테이프를 역전파)

    roll 은 테이프 없는 전진에 쓰는 작업 공간이다. 최적화 반복 전체가 하나를
    돌려 쓴다.

    return: (loss, pos_final)
    """
    n = ptl_type.shape[0]
    inv_n = 1.0 / float(n_ptl)

    tape0 = wp.Tape()
    pos0 = ptl_place(cfg, base_pos, offset, ptl_type, tape=tape0)    # [#all, 3]
    vel0 = wp.zeros(n, dtype=wp.vec3, requires_grad=True)            # [#all, 3]

    pos_final, _, _ = sim.simulate(cfg, pos0, vel0, ptl_type, n_steps, roll=roll)
    loss = loss_value(pos_final, pos_target, ptl_type, inv_n)

    seed_gpos = wp.zeros(n, dtype=wp.vec3)                           # [#all, 3]
    wp.launch(kn.loss_seed_cal, dim=n,
              inputs=[pos_final, pos_target, ptl_type, inv_n], outputs=[seed_gpos])
    seed_gvel = wp.zeros(n, dtype=wp.vec3)                           # [#all, 3]

    g_pos0, _ = ckpt.backward_rollout(
        cfg, pos0, vel0, ptl_type, n_steps, 0, seed_gpos, seed_gvel, cfg.ckpt_depth, roll
    )

    # offset 은 최적화 반복 사이에 재사용하는 배열이다. Tape.backward 는 .grad 에
    # 누적하므로 반드시 먼저 0 으로 지운다. 지우지 않으면 매 반복의 그래디언트가
    # 조용히 더해져서 최적화가 엉뚱한 방향으로 간다.
    offset.grad.zero_()
    tape0.backward(grads={pos0: g_pos0})
    return loss, pos_final


def optimization(
    cfg: Config,
    base_pos: wp.array,         # [#all, 3]
    ptl_type: wp.array,         # [#all]
    n_ptl: int,
    pos_target: wp.array,       # [#all, 3]
    verbose: bool = True,
) -> tuple[wp.array, list[dict[str, Any]]]:
    """최적화 루프. 반복마다 기록한 프레임 정보를 함께 돌려준다.

    cfg: 설정값 묶음
    base_pos: offset 을 더하기 전의 기준 위치
    ptl_type: 입자 종류
    n_ptl: 유체 입자 수
    pos_target: 목표 최종 상태
    verbose: 반복마다 진행 상황을 찍을지

    return: (offset, history)
    """
    offset = offset_array(cfg.init_offset_x, cfg.init_offset_y)
    opt = Adam([offset], lr=cfg.opt_lr)
    roll = sim.Rollout(cfg, ptl_type.shape[0])   # 반복 전체가 버퍼를 공유한다
    true_off = np.array([cfg.true_offset_x, cfg.true_offset_y])

    history: list[dict[str, Any]] = []
    for it in range(cfg.opt_steps):
        opt.lr = cfg.opt_lr * (cfg.opt_lr_decay ** it)
        loss, pos_final = loss_and_grad(
            cfg, offset, base_pos, ptl_type, pos_target, n_ptl, cfg.opt_sim_steps, roll
        )
        g = offset.grad.numpy()[0].copy()
        o = offset.numpy()[0].copy()

        pos0_now = ptl_place(cfg, base_pos, offset, ptl_type)
        history.append({
            "iter": it,
            "loss": loss,
            "offset": o[:2].copy(),
            "grad": g[:2].copy(),
            "pos0": pos0_now.numpy().copy(),            # [#all, 3]
            "pos_final": pos_final.numpy().copy(),      # [#all, 3]
        })

        if verbose:
            err = float(np.linalg.norm(o[:2] - true_off))
            print(f"  iter {it:3d}  loss {loss:.6e}  offset ({o[0]:+.5f}, {o[1]:+.5f})  "
                  f"|offset-true| {err:.5f}  grad ({g[0]:+.4e}, {g[1]:+.4e})")

        opt.step([offset.grad])
        clamp_offset(cfg, offset)

    return offset, history
