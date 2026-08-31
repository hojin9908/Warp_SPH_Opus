"""결과 저장. matplotlib 으로 프레임을 그리고 PIL 로 GIF 를 만든다.

궤적은 Reference 코드와 같은 포맷으로도 저장한다.
    simulation_trajectory.npy   # [time, #ptl, 5(x, y, vx, vy, m)]
    boundary.npy                # [#bnd, 5(x, y, vx, vy, m)]
이 두 파일은 Reference/visualize/animate.py 와 statistic 도구가 그대로 읽는다.
"""

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from source.config import Config
from source.gen_ptl import PTL

# 어두운 배경 위의 물. 느린 곳은 짙은 남색, 빠른 선단은 밝은 하늘색이 된다.
BACKGROUND = "#0b1220"
WALL_COLOR = "#2b3650"
WATER_CMAP = LinearSegmentedColormap.from_list(
    "water", ["#12306b", "#1d6fc0", "#35c6e8", "#d9f7ff"]
)


def figure_to_image(fig: "matplotlib.figure.Figure") -> Image.Image:
    """matplotlib figure 를 PIL 이미지로 바꾼다."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[:, :, :3].copy())


def save_gif(images: list[Image.Image], path: str, fps: int) -> None:
    """프레임 리스트를 GIF 한 장으로 저장한다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # 적응 팔레트로 양자화해서 색 띠가 생기지 않게 하고 파일 크기도 줄인다.
    frames = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in images]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / max(fps, 1)),
        loop=0,
        optimize=True,
    )
    print(f"  저장: {path}  ({len(images)} 프레임)")


def plot_limits(cfg: Config, ytop: float) -> tuple[float, float, float, float]:
    """그림의 x, y 범위를 정한다."""
    m = 4 * cfg.dx
    return (-cfg.bnd_layer * cfg.dx - m, cfg.tank_width + cfg.bnd_layer * cfg.dx + m,
            -cfg.bnd_layer * cfg.dx - m, ytop)


def save_trajectory(
    cfg: Config,
    snaps: list[tuple[np.ndarray, np.ndarray]],
    ptl_type: np.ndarray,       # [#all]
    out_dir: str,
) -> None:
    """궤적을 Reference 포맷 .npy 로 저장한다.

    cfg: 설정값 묶음
    snaps: (pos, vel) 스냅샷 리스트. 각각 [#all, 3]
    ptl_type: 입자 종류
    out_dir: 저장 폴더
    """
    os.makedirs(out_dir, exist_ok=True)
    is_ptl = ptl_type == PTL
    is_bnd = ~is_ptl
    n_time = len(snaps)
    n_ptl = int(is_ptl.sum())

    traj = np.zeros((n_time, n_ptl, 5), dtype=np.float32)       # [time, #ptl, 5]
    for t, (pos, vel) in enumerate(snaps):
        traj[t, :, 0:2] = pos[is_ptl, :2]
        traj[t, :, 2:4] = vel[is_ptl, :2]
        traj[t, :, 4] = cfg.mass

    pos0 = snaps[0][0]
    bnd = np.zeros((int(is_bnd.sum()), 5), dtype=np.float32)    # [#bnd, 5]
    bnd[:, 0:2] = pos0[is_bnd, :2]
    bnd[:, 4] = cfg.mass

    traj_path = os.path.join(out_dir, "simulation_trajectory.npy")
    bnd_path = os.path.join(out_dir, "boundary.npy")
    np.save(traj_path, traj)
    np.save(bnd_path, bnd)
    print(f"  저장: {traj_path}  {traj.shape}")
    print(f"  저장: {bnd_path}  {bnd.shape}")


def forward_animation(
    cfg: Config,
    snaps: list[tuple[np.ndarray, np.ndarray]],
    ptl_type: np.ndarray,       # [#all]
    path: str,
) -> None:
    """순방향 시뮬레이션 GIF. 유체는 속도 크기로 색을 준다.

    축과 눈금을 지우고 어두운 배경에 물 색 계열을 써서 결과 자체가 보이게 한다.
    그림 비율은 계산 영역의 비율에 맞춰 잡아 여백을 남기지 않는다.
    """
    is_ptl = ptl_type == PTL
    is_bnd = ~is_ptl
    n_ptl = int(is_ptl.sum())

    # 튀어나간 물방울 몇 개가 색 범위를 독차지하지 않도록 99 퍼센타일로 자른다.
    speeds = np.concatenate([np.linalg.norm(v[is_ptl, :2], axis=1) for _, v in snaps])
    vmax = max(float(np.percentile(speeds, 99.0)), 1e-6)
    ytop = float(np.percentile(np.concatenate([p[is_ptl, 1] for p, _ in snaps]), 99.8))
    ytop = min(max(ytop + 3 * cfg.dx, cfg.tank_height * 0.5), cfg.tank_height * 0.8)
    x0, x1, y0, y1 = plot_limits(cfg, ytop)

    # 계산 영역 비율에 맞춰 그림 크기와 축 위치를 직접 정한다 (tight_layout 없이).
    # 글자는 축 위쪽 여백에 두어 유체와 겹치지 않게 한다.
    fig_w = 9.0
    ax_x, ax_w, ax_frac = 0.015, 0.895, 0.845
    ax_h_in = fig_w * ax_w * (y1 - y0) / (x1 - x0)
    fig_h = ax_h_in / ax_frac

    # 입자 하나가 화면에서 차지할 지름(pt)을 계산해 scatter 크기로 쓴다.
    pt_per_data = fig_w * ax_w * 72.0 / (x1 - x0)
    ptl_size = (1.35 * cfg.dx * pt_per_data) ** 2

    images: list[Image.Image] = []
    for i, (pos, vel) in enumerate(snaps):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=cfg.dpi)
        fig.patch.set_facecolor(BACKGROUND)
        ax = fig.add_axes((ax_x, 0.02, ax_w, ax_frac))
        ax.set_facecolor(BACKGROUND)

        speed = np.linalg.norm(vel[is_ptl, :2], axis=1)
        # 경계는 정사각형을 빈틈없이 붙여 벽처럼 보이게 한다.
        ax.scatter(pos[is_bnd, 0], pos[is_bnd, 1], s=ptl_size * 1.15, c=WALL_COLOR,
                   marker="s", linewidths=0)
        sc = ax.scatter(pos[is_ptl, 0], pos[is_ptl, 1], s=ptl_size,
                        c=np.minimum(speed, vmax), cmap=WATER_CMAP,
                        vmin=0.0, vmax=vmax, linewidths=0)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.text(ax_x + 0.004, 0.965, f"t = {i * cfg.frame_step * cfg.dt:5.3f} s",
                 color="#e8f4ff", fontsize=15, family="monospace", va="top")
        fig.text(ax_x + ax_w, 0.955,
                 f"2D WCSPH   {n_ptl} fluid + {int(is_bnd.sum())} boundary   "
                 f"{cfg.kernel_type} kernel   h/dx = {cfg.h / cfg.dx:.2f}",
                 color="#7f8fb0", fontsize=9, family="monospace",
                 va="top", ha="right")

        cax = fig.add_axes((ax_x + ax_w + 0.012, 0.12, 0.011, 0.55))
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("|v|  [m/s]", color="#9fb2d0", fontsize=8.5)
        cb.ax.tick_params(colors="#9fb2d0", labelsize=7.5, length=2)
        cb.outline.set_visible(False)

        images.append(figure_to_image(fig))
        plt.close(fig)

    save_gif(images, path, cfg.fps)


def optimize_animation(
    cfg: Config,
    history: list[dict[str, Any]],
    ptl_type: np.ndarray,       # [#all]
    pos_target: np.ndarray,     # [#all, 3]
    path: str,
) -> None:
    """최적화 과정 GIF. 왼쪽은 초기 블록, 오른쪽은 최종 상태와 목표의 비교."""
    is_ptl = ptl_type == PTL
    is_bnd = ~is_ptl
    ytop = max(float(pos_target[is_ptl, 1].max()),
               float(history[0]["pos0"][is_ptl, 1].max())) + 6 * cfg.dx
    x0, x1, y0, y1 = plot_limits(cfg, ytop)
    true_off = np.array([cfg.true_offset_x, cfg.true_offset_y])

    images: list[Image.Image] = []
    for h in history:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=cfg.dpi)

        ax = axes[0]
        ax.scatter(h["pos0"][is_bnd, 0], h["pos0"][is_bnd, 1], s=3, c="0.7",
                   marker="s", linewidths=0)
        ax.scatter(h["pos0"][is_ptl, 0], h["pos0"][is_ptl, 1], s=6, c="tab:green",
                   linewidths=0, label="current initial block")
        ax.scatter(h["pos0"][is_ptl, 0] - h["offset"][0] + true_off[0],
                   h["pos0"][is_ptl, 1] - h["offset"][1] + true_off[1],
                   s=6, facecolors="none", edgecolors="tab:red", linewidths=0.4,
                   label="target initial block")
        ax.set_title(f"initial state   offset = ({h['offset'][0]:+.4f}, {h['offset'][1]:+.4f})"
                     f"   true = ({true_off[0]:+.4f}, {true_off[1]:+.4f})")
        ax.legend(loc="upper right", fontsize=7)

        ax2 = axes[1]
        ax2.scatter(pos_target[is_bnd, 0], pos_target[is_bnd, 1], s=3, c="0.7",
                    marker="s", linewidths=0)
        ax2.scatter(pos_target[is_ptl, 0], pos_target[is_ptl, 1], s=8, facecolors="none",
                    edgecolors="tab:red", linewidths=0.4, label="target final")
        ax2.scatter(h["pos_final"][is_ptl, 0], h["pos_final"][is_ptl, 1], s=5,
                    c="tab:blue", linewidths=0, label="current final")
        ax2.set_title(f"final state after {cfg.opt_sim_steps} steps   "
                      f"iter {h['iter']}   loss = {h['loss']:.4e}")
        ax2.legend(loc="upper right", fontsize=7)

        for a in axes:
            a.set_xlim(x0, x1)
            a.set_ylim(y0, y1)
            a.set_aspect("equal")

        fig.tight_layout()
        images.append(figure_to_image(fig))
        plt.close(fig)

    save_gif(images, path, max(cfg.fps // 4, 2))


def loss_curve(losses: list[float], path: str) -> None:
    """최적화 손실 곡선을 PNG 로 저장한다."""
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=120)
    ax.semilogy(losses, marker="o", ms=3)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  저장: {path}")
