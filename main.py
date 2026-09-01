"""미분 가능한 2D WCSPH dam break 실행 진입점.

    python main.py --mode forward      # 순방향 시뮬레이션 + GIF + 궤적 .npy
    python main.py --mode optimize     # 초기 블록 위치 복원 + GIF
    python main.py --mode check        # 그래디언트 검증
    python main.py --mode all
"""

import argparse
import os
import time
from typing import Any

import numpy as np
import warp as wp

from source import checkpoint as ckpt
from source import optimize as opt
from source import simulation as sim
from source.config import Config
from source.gen_ptl import particle_generation, to_warp
from visualize import animate


def str2bool(v: str) -> bool:
    """argparse 용 bool 파서. type=bool 은 "False" 도 True 로 만들어 버린다."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"bool 값이 아니다: {v}")


def parsing() -> dict[str, Any]:
    """명령행 인자를 읽는다. mode 와 dump_config 를 뺀 나머지는 Config 필드와 1:1 이다."""
    parser = argparse.ArgumentParser(description="Differentiable 2D WCSPH dam break with Warp")

    # run property
    parser.add_argument("--mode", type=str, choices=["forward", "optimize", "check", "all"],
                        help="what to run", default="all")
    parser.add_argument("--dump_config", type=str, help="path to save the resolved config JSON",
                        default=None)

    # project property
    parser.add_argument("--device", type=str, help="device to use", default="cuda:0")
    parser.add_argument("--out_dir", type=str, help="directory to save results",
                        default="outputs")
    parser.add_argument("--seed", type=int, help="random seed", default=123)

    # simulation setting
    parser.add_argument("--dx", type=float, help="particle spacing", default=0.02)
    parser.add_argument("--tank_width", type=float, help="width of the dam", default=2.0)
    parser.add_argument("--tank_height", type=float, help="height of the dam", default=1.0)
    parser.add_argument("--fluid_width", type=float,
                        help="width of initial particle block", default=0.5)
    parser.add_argument("--fluid_height", type=float,
                        help="height of initial particle block", default=0.5)
    parser.add_argument("--fluid_origin_x", type=float,
                        help="x of the block's lower-left corner", default=0.0)
    parser.add_argument("--fluid_origin_y", type=float,
                        help="y of the block's lower-left corner", default=0.0)
    parser.add_argument("--bnd_layer", type=int,
                        help="number of particle layer for dam boundary", default=3)
    parser.add_argument("--jitter", type=float,
                        help="random jitter on initial position [dx]", default=0.0)

    # physical coefficient
    parser.add_argument("--rho0", type=float, help="reference density", default=1000.0)
    parser.add_argument("--gamma", type=float,
                        help="stiffness parameter (1<=gamma<=7)", default=7.0)
    parser.add_argument("--c0", type=float,
                        help="speed of sound, 0 for 10*sqrt(2*g*H)", default=0.0)
    parser.add_argument("--mu", type=float, help="dynamic viscosity [Pa s]", default=0.05)
    parser.add_argument("--g", type=float, help="gravitational acceleration", default=9.81)
    parser.add_argument("--h", type=float,
                        help="smoothing length, 0 for h_factor*dx", default=0.0)
    parser.add_argument("--h_factor", type=float,
                        help="h = h_factor * dx when --h is 0", default=1.3)
    parser.add_argument("--kernel_type", type=str, choices=["cubic", "wendland"],
                        help="SPH kernel", default="cubic")
    parser.add_argument("--clamp_negative_pressure", type=str2bool,
                        help="clamp negative pressure to zero", default=True)


    # density filter
    parser.add_argument("--shepard", type=str2bool,
                        help="whether use Shepard filter for density calculation", default=True)
    parser.add_argument("--shepard_step", type=int,
                        help="step interval for applying shepard filter", default=20)

    # PDE solver hyperparameter
    parser.add_argument("--dt", type=float,
                        help="time step, 0 for cfl*h/c0", default=0.0)
    parser.add_argument("--cfl", type=float, help="CFL number for automatic dt", default=0.25)
    parser.add_argument("--n_steps", type=int,
                        help="number of steps for forward simulation", default=4000)
    parser.add_argument("--frame_step", type=int,
                        help="step interval between animation frames", default=40)

    # hash grid
    parser.add_argument("--grid_dim", type=int,
                        help="hash bucket count per axis", default=128)

    # recursive checkpointing
    parser.add_argument("--ckpt_depth", type=int,
                        help="recursive checkpointing depth r", default=2)
    parser.add_argument("--ckpt_min_segment", type=int,
                        help="segments shorter than this go on one tape", default=8)

    # optimization
    parser.add_argument("--opt_steps", type=int,
                        help="number of optimization iterations", default=60)
    parser.add_argument("--opt_sim_steps", type=int,
                        help="simulation length used in optimization", default=400)
    parser.add_argument("--opt_lr", type=float, help="Adam learning rate", default=0.01)
    parser.add_argument("--opt_lr_decay", type=float,
                        help="learning rate decay per iteration", default=0.95)
    parser.add_argument("--true_offset_x", type=float,
                        help="ground-truth initial offset x", default=0.10)
    parser.add_argument("--true_offset_y", type=float,
                        help="ground-truth initial offset y", default=0.05)
    parser.add_argument("--init_offset_x", type=float,
                        help="optimization start offset x", default=0.02)
    parser.add_argument("--init_offset_y", type=float,
                        help="optimization start offset y", default=0.01)

    # animation setting
    parser.add_argument("--fps", type=int, help="frame per second for animation", default=20)
    parser.add_argument("--dpi", type=int, help="dpi for animation frames", default=80)
    parser.add_argument("--save_npy", type=str2bool,
                        help="save trajectory as .npy in Reference format", default=True)

    args = vars(parser.parse_args())

    return args


def setup(cfg: Config) -> tuple[wp.array, wp.array, np.ndarray, int]:
    """디바이스를 잡고 초기 입자를 만든다.

    return: (base_pos, ptl_type, ptl_type_numpy, n_ptl)
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    wp.init()
    wp.set_device(cfg.device)
    print("=" * 78)
    print(cfg.summary())
    print("=" * 78)

    gen = particle_generation(cfg)
    base_np, ptl_type_np, n_ptl = gen.build()
    base_pos, ptl_type = to_warp(base_np, ptl_type_np)
    print(f"입자 {len(ptl_type_np)} 개  "
          f"(유체 {n_ptl}, 경계 {len(ptl_type_np) - n_ptl})")
    return base_pos, ptl_type, ptl_type_np, n_ptl


def run_forward(cfg: Config, base_pos: wp.array, ptl_type: wp.array,
                ptl_type_np: np.ndarray) -> None:
    """순방향 시뮬레이션을 돌리고 GIF 와 궤적을 저장한다."""
    print(f"\n[forward] {cfg.n_steps} 스텝  (t = {cfg.n_steps * cfg.dt:.3f} s)")
    n = ptl_type.shape[0]
    # 순방향 데모는 유체 블록을 왼쪽 벽에 붙여 놓은 고전적인 dam break 다.
    offset = opt.offset_array(0.0, 0.0)
    pos0 = opt.ptl_place(cfg, base_pos, offset, ptl_type)
    vel0 = wp.zeros(n, dtype=wp.vec3)

    t0 = time.time()
    _, _, snaps = sim.simulate(cfg, pos0, vel0, ptl_type, cfg.n_steps,
                               snapshot_step=cfg.frame_step)
    wp.synchronize()
    wall = time.time() - t0
    print(f"  {wall:.2f} s  ({cfg.n_steps / wall:.0f} steps/s)")

    animate.forward_animation(cfg, snaps, ptl_type_np,
                              os.path.join(cfg.out_dir, "forward.gif"))
    if cfg.save_npy:
        animate.save_trajectory(cfg, snaps, ptl_type_np, cfg.out_dir)


def run_optimize(cfg: Config, base_pos: wp.array, ptl_type: wp.array,
                 ptl_type_np: np.ndarray, n_ptl: int) -> None:
    """목표 상태를 만들고 초기 블록 위치를 복원한다."""
    print(f"\n[optimize] 시뮬레이션 {cfg.opt_sim_steps} 스텝 x 최적화 {cfg.opt_steps} 회")
    print("checkpoint 구성:")
    print(ckpt.checkpoint_report(cfg, cfg.opt_sim_steps))

    n = ptl_type.shape[0]
    # 목표 상태: 참값 offset 으로 돌린 결과
    off_true = opt.offset_array(cfg.true_offset_x, cfg.true_offset_y)
    pos0_true = opt.ptl_place(cfg, base_pos, off_true, ptl_type)
    vel0 = wp.zeros(n, dtype=wp.vec3)
    pos_target, _, _ = sim.simulate(cfg, pos0_true, vel0, ptl_type, cfg.opt_sim_steps)
    print(f"목표 상태 생성 완료 (true offset = "
          f"{cfg.true_offset_x:+.4f}, {cfg.true_offset_y:+.4f})\n")

    t0 = time.time()
    offset, history = opt.optimization(cfg, base_pos, ptl_type, n_ptl, pos_target)
    wp.synchronize()
    print(f"\n  {time.time() - t0:.1f} s")

    o = offset.numpy()[0]
    err = float(np.linalg.norm(o[:2] - np.array([cfg.true_offset_x, cfg.true_offset_y])))
    print(f"  복원된 offset ({o[0]:+.5f}, {o[1]:+.5f})  "
          f"참값 ({cfg.true_offset_x:+.5f}, {cfg.true_offset_y:+.5f})  오차 {err:.5f}")

    animate.optimize_animation(cfg, history, ptl_type_np, pos_target.numpy(),
                               os.path.join(cfg.out_dir, "optimize.gif"))
    animate.loss_curve([h["loss"] for h in history],
                       os.path.join(cfg.out_dir, "loss_curve.png"))


def main() -> None:
    args = parsing()
    mode = args.pop("mode")
    dump_config = args.pop("dump_config")

    cfg = Config(**args)
    print("=============== Simulation start ===============")
    if dump_config:
        os.makedirs(os.path.dirname(dump_config) or ".", exist_ok=True)
        cfg.save(dump_config)
        print(f"설정 저장: {dump_config}")

    base_pos, ptl_type, ptl_type_np, n_ptl = setup(cfg)

    if mode in ("check", "all"):
        from statistic import check_grad
        print("\n[check] 그래디언트 검증")
        check_grad.run_check(cfg)
    if mode in ("forward", "all"):
        run_forward(cfg, base_pos, ptl_type, ptl_type_np)
    if mode in ("optimize", "all"):
        run_optimize(cfg, base_pos, ptl_type, ptl_type_np, n_ptl)


if __name__ == "__main__":
    main()
