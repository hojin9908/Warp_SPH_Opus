import json
import math
from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Config:
    """모든 설정값을 한 곳에 모아 둔다.

    main.py 의 argparse 플래그 이름이 이 클래스의 필드 이름과 1:1 로 같다.
    그래서 Config(**args) 한 줄로 만들어진다.
    h, 지지 반경, 질량, Tait 상수 B, dt, CUDA graph 블록 길이 같은 파생값은
    __post_init__ 에서 계산한다.

    device: 계산에 쓸 디바이스 ("cuda:0", "cpu" ...)
    out_dir: 결과 저장 폴더
    seed: 초기 배치 흔들림용 난수 시드
    dx: 입자 간격
    tank_width: 수조 내부 폭
    tank_height: 수조 내부 높이 (위쪽은 열려 있다)
    fluid_width: 초기 유체 블록 폭
    fluid_height: 초기 유체 블록 높이
    fluid_origin_x: 유체 블록 왼쪽 아래 모서리 x
    fluid_origin_y: 유체 블록 왼쪽 아래 모서리 y
    bnd_layer: dummy boundary particle 겹 수
    jitter: 초기 위치에 더할 무작위 흔들림 (dx 배수)
    rho0: 기준 밀도
    gamma: Tait 지수 (1<=gamma<=7)
    c0: 수치 음속. 0 이면 10*sqrt(2*g*fluid_height) 로 자동 계산
    mu: 점성계수 [Pa s]
    g: 중력 가속도
    h: smoothing length. 0 이면 h_factor * dx 로 자동 계산
    h_factor: h 자동 계산에 쓰는 배수 (h = h_factor * dx)
    kernel_type: SPH 커널 종류 ("cubic" 또는 "wendland")
    clamp_negative_pressure: 음압을 0 으로 자를지 (인장 불안정 방지)
    shepard: Shepard filter 사용 여부
    shepard_step: Shepard filter 적용 주기 [step]
    dt: 시간 간격. 0 이면 cfl * h / c0 로 자동 계산
    cfl: dt 자동 계산에 쓰는 CFL 수
    n_steps: 순방향 시뮬레이션 스텝 수
    frame_step: GIF 프레임을 남길 스텝 간격
    grid_dim: HashGrid 해시 버킷 한 변의 개수
    use_cuda_graph: 테이프 없는 전진 구간을 CUDA graph 로 묶을지
    graph_min_steps: 그래프 하나가 담을 최소 스텝 수
    graph_in_grad: 그래디언트 경로에서도 그래프를 쓸지 (README 5절 참고)
    ckpt_depth: recursive checkpointing 깊이 r
    ckpt_min_segment: 이보다 짧은 구간은 통째로 테이프에 올린다
    opt_steps: 최적화 반복 횟수
    opt_sim_steps: 최적화에 쓰는 시뮬레이션 길이
    opt_lr: Adam 학습률
    opt_lr_decay: 반복마다 학습률에 곱하는 값
    true_offset_x: 목표 상태를 만든 실제 초기 이동량 x
    true_offset_y: 목표 상태를 만든 실제 초기 이동량 y
    init_offset_x: 최적화 시작점 x
    init_offset_y: 최적화 시작점 y
    fps: GIF 프레임 레이트
    dpi: GIF 해상도
    save_npy: 궤적을 .npy 로 저장할지
    """

    # project property
    device: str = "cuda:0"
    out_dir: str = "outputs"
    seed: int = 123

    # simulation setting
    dx: float = 0.02
    tank_width: float = 2.0
    tank_height: float = 1.0
    fluid_width: float = 0.5
    fluid_height: float = 0.5
    fluid_origin_x: float = 0.0
    fluid_origin_y: float = 0.0
    bnd_layer: int = 3
    jitter: float = 0.0

    # physical coefficient
    rho0: float = 1000.0
    gamma: float = 7.0
    c0: float = 0.0
    mu: float = 0.05
    g: float = 9.81
    h: float = 0.0
    h_factor: float = 1.3
    kernel_type: str = "cubic"
    clamp_negative_pressure: bool = True

    # density filter
    shepard: bool = True
    shepard_step: int = 20

    # PDE solver hyperparameter
    dt: float = 0.0
    cfl: float = 0.25
    n_steps: int = 4000
    frame_step: int = 40

    # hash grid
    grid_dim: int = 128

    # CUDA graph
    use_cuda_graph: bool = True
    graph_min_steps: int = 20
    graph_in_grad: bool = False

    # recursive checkpointing
    ckpt_depth: int = 2
    ckpt_min_segment: int = 8

    # optimization
    opt_steps: int = 60
    opt_sim_steps: int = 400
    opt_lr: float = 0.01
    opt_lr_decay: float = 0.95
    true_offset_x: float = 0.10
    true_offset_y: float = 0.05
    init_offset_x: float = 0.02
    init_offset_y: float = 0.01

    # animation setting
    fps: int = 20
    dpi: int = 80
    save_npy: bool = True

    def __post_init__(self) -> None:
        # h 는 입력 파라미터다. 0 이면 입자 간격으로부터 자동 계산한다.
        self.auto_h: bool = self.h <= 0.0
        if self.auto_h:
            self.h = self.h_factor * self.dx
        self.support: float = 2.0 * self.h                      # 커널 지지 반경 = 이웃 탐색 반경
        self.mass: float = self.rho0 * self.dx * self.dx        # 2D 입자 질량

        if self.kernel_type not in ("cubic", "wendland"):
            raise ValueError(f"kernel_type 은 'cubic' 또는 'wendland' 여야 한다: {self.kernel_type}")
        self.kernel_id: int = 0 if self.kernel_type == "cubic" else 1

        # c0 와 dt 도 0 이면 자동 계산이다. 어느 쪽이었는지 기억해 두었다가
        # to_dict() 에서 다시 0 으로 되돌린다.
        self.auto_c0: bool = self.c0 <= 0.0
        self.auto_dt: bool = self.dt <= 0.0

        if self.c0 <= 0.0:
            self.c0 = 10.0 * math.sqrt(2.0 * self.g * max(self.fluid_height, 1e-6))
        self.B: float = self.rho0 * self.c0 * self.c0 / self.gamma   # Tait 상수

        if self.dt <= 0.0:
            self.dt = self.cfl * self.h / self.c0

        # 최적화 중 유체 블록이 수조 밖으로 나가지 않도록 offset 을 가두는 범위.
        # 벽 입자 위로 올라타면 물리가 깨져서 그래디언트가 의미를 잃는다.
        self.offset_lo: tuple[float, float] = (0.0 - self.fluid_origin_x,
                                               0.0 - self.fluid_origin_y)
        self.offset_hi: tuple[float, float] = (
            self.tank_width - self.fluid_width - self.fluid_origin_x,
            0.5 * self.tank_height - self.fluid_origin_y,
        )

        # CUDA graph 한 덩어리의 스텝 수.
        #  - Shepard 주기의 배수여야 그래프 안의 분기 패턴이 매 블록 같다.
        #  - 짝수여야 ping-pong 버퍼가 그래프 끝에서 제자리로 돌아온다.
        period = self.shepard_step if (self.shepard and self.shepard_step > 0) else 1
        block = period * max(1, math.ceil(self.graph_min_steps / period))
        self.graph_block: int = block * 2 if block % 2 == 1 else block

    def to_dict(self) -> dict[str, Any]:
        """dataclass 필드만 dict 로 돌려준다. 자동 계산이었던 값은 0 인 채로 남긴다."""
        d = asdict(self)
        if self.auto_h:
            d["h"] = 0.0         # 자동 계산이었던 값은 자동인 채로 저장한다
        if self.auto_c0:
            d["c0"] = 0.0
        if self.auto_dt:
            d["dt"] = 0.0
        return d

    def save(self, path: str) -> None:
        """설정을 JSON 으로 저장한다."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """실행할 때 맨 위에 찍을 요약 문자열."""
        return (
            f"device={self.device}  dx={self.dx}  "
            f"h={self.h:.5f}{' (auto)' if self.auto_h else ' (given)'}  "
            f"support={self.support:.5f}  kernel={self.kernel_type}\n"
            f"mass={self.mass:.5f}  rho0={self.rho0}  c0={self.c0:.3f}  B={self.B:.1f}  "
            f"gamma={self.gamma}  mu={self.mu}\n"
            f"dt={self.dt:.3e}  n_steps={self.n_steps}  shepard={self.shepard}"
            f"(every {self.shepard_step})  bnd_layer={self.bnd_layer}\n"
            f"ckpt_depth={self.ckpt_depth}  ckpt_min_segment={self.ckpt_min_segment}\n"
            f"use_cuda_graph={self.use_cuda_graph}  graph_block={self.graph_block}"
            f"  graph_in_grad={self.graph_in_grad}"
        )
