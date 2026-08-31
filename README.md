# Warp SPH — 미분 가능한 2D WCSPH Dam Break

JAX 없이 **NVIDIA Warp 만으로** 쓴 2D WCSPH dam break 시뮬레이터다.
이웃 탐색은 Warp 내장 `HashGrid` 를 매 스텝 갱신해 동적으로 하고, 자동미분은
`wp.Tape` 만 쓰며, 긴 시뮬레이션의 메모리는 **직접 구현한 recursive
checkpoint/replay** 로 줄인다. smoothing length 는 고정과 가변을, 커널은 cubic
spline 과 Wendland C2 를 모두 지원하고, 테이프 없는 전진 구간은 **CUDA graph** 로
묶어 커널 런치 오버헤드를 없앤다.

폴더 구조·이름 규칙·궤적 저장 포맷은 `Reference/` 의 기존 SPH 코드에 맞췄다.

---

## 1. 실행

```bash
python main.py --mode forward      # 순방향 시뮬레이션 + GIF + 궤적 .npy
```
```bash
python main.py --mode optimize     # 초기 블록 위치 복원 + GIF
```
```bash
python main.py --mode check        # 그래디언트 검증 (유한차분 대조 + 카오스 진단)
```
```bash
python main.py --mode all
```

설정은 전부 명령행 플래그다. 플래그 이름이 `Config` 의 필드 이름과 1:1 로 같아서
`Config(**args)` 한 줄로 만들어진다. `--help` 로 전체 목록을 볼 수 있다.

```bash
python main.py --mode optimize --device cuda:1 --h 0.03 --opt_sim_steps 200
```
```bash
python main.py --mode forward --kernel_type wendland --dx 0.01 --n_steps 8000
```
```bash
python main.py --mode all --dump_config outputs/config_used.json
```

`--dump_config` 는 실제로 쓰인 설정을 JSON 으로 남긴다. `c0` 와 `dt` 는 0 이면
자동 계산이고(`c0 = 10√(2gH)`, `dt = cfl·h/c0`), 저장할 때도 0 인 채로 남는다.
그래서 저장한 설정에 `dx` 만 바꿔 다시 실행해도 `dt` 가 새 해상도에 맞춰 다시
계산된다. 실제로 쓰인 값은 실행할 때마다 맨 위 요약에 찍힌다.

> argparse 를 쓴 것은 지금 단계에서 가독성이 우선이기 때문이다. MCP 로 제어할
> 때가 되면 `Config` 를 그대로 두고 입력 계층만 바꾸면 된다.

## 2. 파일

```
main.py                     CLI (argparse) 와 실행 흐름
source/config.py            모든 설정값과 파생값 (h, 탐색 반경, mass, B, dt, graph_block)
source/gen_ptl.py           particle_generation — 초기 유체 블록과 dam 경계
source/kernel.py            SPH 커널 함수와 Warp kernel 전부
source/simulation.py        sph_step (한 스텝), simulate (전진), Rollout (CUDA graph)
source/checkpoint.py        recursive checkpoint / replay 역전파
source/optimize.py          loss, 그래디언트, Adam 최적화 루프
visualize/animate.py        GIF 저장과 궤적 .npy 저장
statistic/check_grad.py     유한차분 대조와 궤적 발산 측정
```

배열 표기는 `# [#all, 3]` 처럼 주석으로 붙였다. `#all = #ptl + #bnd` 이고
유체 입자와 경계 입자를 한 배열에 담는다 (HashGrid 를 하나만 쓰기 위해서다).

## 3. 한 스텝

`simulation.sph_step()` 하나만 읽으면 전부 보인다.

```
grid.build(pos, support)       # 이웃 탐색  ← 반드시 tape 밖
  ↓
density_cal                    # 밀도   rho = m Σ W(r, h)
  ↓
shepard_filter | wp.copy       # 밀도 보정 (shepard_step 스텝마다)
  ↓
pres_cal                       # 압력   Tait  p = B((rho/rho0)^gamma - 1)
  ↓
force_cal                      # 힘     압력 + 점성(Morris) + 중력
  ↓
vel_pos_step                   # 적분   semi-implicit Euler, 경계는 고정
```

압력력·점성력·중력을 커널 하나에 묶은 이유는 이웃 순회를 한 번만 돌기 위해서다.
조밀행렬 구현이라면 나눠도 공짜지만, HashGrid 에서는 나눌 때마다 탐색이 한 벌씩 는다.
속도와 위치도 한 커널인데, 위치가 갱신된 속도를 써야 하기 때문이다.

### 물리

- **커널**: cubic spline 또는 Wendland C2, 둘 다 지지 반경 `2h`.
  `q = 2` 에서 `W = 0` 이고 `dW/dr = 0` 이라 `if r < 2h` 컷오프를 넣어도
  자동미분이 델타 항을 놓치지 않는다. 커널을 바꿀 때는 이 성질을 먼저 확인해야 한다.
- **밀도**: 커널 합. 자기 자신 항은 `W(0)` 상수로 더하고 이웃 루프에서는
  `r² > 1e-12` 로 제외한다 (`wp.sqrt(0)` 의 adjoint 가 발산한다).
- **밀도 보정**: Shepard filter `rho ← Σ m W / Σ (m/rho_j) W`.
  `shepard_step` 스텝마다 적용하고, 적용하지 않는 스텝은 `wp.copy` 로 넘긴다
  (`wp.copy` 는 미분 가능하므로 테이프가 끊기지 않는다).
- **압력**: Tait, `B = rho0 c0² / gamma`, `c0 = 10 √(2 g H)`.
- **힘**: 대칭형 압력력 `-m (p_i/ρ_i² + p_j/ρ_j²) ∇W`, Morris 점성, 그리고 중력.
  둘 다 이웃 루프에서 **순수 누적(`+=`)** 만 하므로 동적 루프에서도 adjoint 가 정확하다.
- **경계**: dummy boundary particle. 기본 3겹(`--bnd_layer`), 바닥과 좌우 벽,
  위는 열려 있다. 밀도 합에는 들어가지만 적분에서는 움직이지 않는다.
  속도가 0 이라 점성항이 그대로 no-slip 벽이 된다.

### 2D 인데 `wp.vec3` 를 쓰는 이유

Warp `HashGrid` 는 3D 점만 받는다. z 를 항상 0 으로 두면
`grid.build(points=pos, radius=...)` 를 그대로 쓸 수 있고, 거리도 2D 거리와
같으므로 물리는 완전히 2D 다 (정규화 상수도 2D 를 쓴다).

## 4. Smoothing length

`h` 는 **초기 입자 배치와 마찬가지로 실행 전에 정해지는 입력**이다.
시뮬레이션 도중에는 바뀌지 않는다.

```bash
python main.py --mode forward --h 0.03          # 직접 지정
python main.py --mode forward --h_factor 1.6    # h = 1.6 * dx 로 자동 계산
python main.py --mode forward                   # 기본 h = 1.3 * dx
```

`--h` 가 0 이면 `h_factor * dx` 로 자동 계산하고, 0 이 아니면 그 값을 그대로 쓴다.
어느 쪽이었는지는 실행 요약에 `(auto)` / `(given)` 으로 찍히고, `--dump_config`
에도 자동이었던 값은 0 인 채로 남는다. 그래서 저장한 설정에 `--dx` 만 바꿔
다시 돌리면 `h` 가 새 해상도에 맞춰 다시 계산된다.

`h` 를 바꾸면 따라 바뀌는 것들:

| | 관계 |
|---|---|
| 커널 지지 반경 `support` | `2h` — HashGrid 셀 크기이자 이웃 탐색 반경 |
| 시간 간격 `dt` | `cfl · h / c0` (`--dt` 로 직접 줄 수도 있다) |
| 이웃 수 | 2D 에서 대략 `π(2h/dx)²` |

`dx` 와 독립적인 입력이라 `h/dx` 비를 바꿔가며 볼 수 있다 (CPU, dx=0.05, 600 스텝):

```
      h  h/dx  support         dt   초기 rho 중앙값   평균 이웃   |v|max   finite
 0.0500  1.00   0.1000  3.991e-04          1000.9        9.0    4.159     True
 0.0650  1.30   0.1300  5.188e-04           999.9       17.9    3.697     True   <- 기본값
 0.0800  1.60   0.1600  6.386e-04          1002.9       31.2    4.108     True
 0.1000  2.00   0.2000  7.982e-04           994.1       38.4    5.316     True
```

`h/dx` 를 키우면 이웃이 늘어 커널 합이 매끄러워지지만 스텝당 비용이 그만큼 는다.
1.0 근처까지 내려도 이웃이 9개라 돌아가기는 하지만, SPH 에서는 보통 1.2\~1.5 를 쓴다.

## 5. SPH 커널 선택

`--kernel_type cubic`(기본) 또는 `--kernel_type wendland`.

| | 정규화 (2D) | 지지 반경 | 자동미분 |
|---|---|---|---|
| cubic spline | `10/(7π h²)` | 2h | 안전 (`W(2h)=0`, `W'(2h)=0`) |
| Wendland C2 | `7/(4π h²)` | 2h | 안전 (`W(2h)=0`, `W'(2h)=0`) |

`Reference/source/kernel.py` 에서 실제로 쓰이는 것이 Wendland C2 라 같은 커널을
넣었다. 커널 종류는 `@wp.func` 안의 정수 분기 하나로 고른다 — 모든 스레드가
같은 값을 보므로 분기 비용이 없다.

> 옮겨오지 않은 것: 같은 파일의 `gauss2d_kernel` 은 `domain = r < 3h` 인데
> `exp(-9) ≈ 1.2e-4` 라 컷오프에서 값이 뚝 끊긴다. 자동미분이 그 델타 항을
> 놓친다. `pres2d_kernel` 은 `(h-r)³` 을 `r < 3h` 까지 써서 `r > h` 에서
> 음수가 된다 (spiky 커널은 보통 `r < h` 에서 자른다).

## 6. CUDA graph

스텝 하나는 커널 런치 5회 + `HashGrid.build`(내부적으로 10여 회)로 이루어진다.
입자가 1,000\~5,000개 규모면 커널 실행 시간보다 **런치 오버헤드가 지배적**이다.
그래서 `graph_block` 개 스텝을 CUDA graph 하나로 캡처해 두고 반복 실행한다.

블록 길이는 자동으로 정해진다.

- **Shepard 주기의 배수**여야 그래프 안의 분기 패턴이 매 블록 같다.
- **짝수**여야 ping-pong 버퍼가 그래프 끝에서 제자리로 돌아온다.

정렬이 맞지 않는 구간은 자동으로 평범한 루프로 떨어진다. 그래프는
`simulation.Rollout` 작업 공간에 캐시되므로, 같은 Rollout 을 여러 `simulate`
호출에 넘기면 캡처는 한 번만 일어난다.

Warp 의 `HashGrid.build` 도 캡처된다 (공식 `example_dem.py` 는 grid build 를
그래프 밖에 두지만, 매 스텝 rebuild 가 요구사항이라 안에 넣고 검증했다).

### 🔴 캡처한 그래프는 다른 HashGrid 를 쓰면 무효가 된다

측정하다 발견했다. 그래프를 캡처한 뒤 **다른 `HashGrid` 를 만들어 build 하고
eager 스텝을 돌리면, 그 다음 graph 실행 결과가 완전히 깨진다.**

```
  캡처만                        |v|max   4.459   유체 x폭   2.013   (정상)
  캡처 후 다른 HashGrid build   |v|max  19.304   유체 x폭  13.602   (유체가 수조 밖으로)
  두 결과 max|diff| 1.205e+01
```

Warp 내부에서 HashGrid 들이 정렬 스크래치를 공유하는 것으로 보인다
(어느 동작이 방아쇠인지 — 생성인지 build 인지 eager 실행인지 — 는 아직
분리하지 못했다. GPU 장애로 중단됐다).

에러도 경고도 없이 조용히 틀린 물리를 돌리므로 코드에 방어를 넣었다.
`simulation.py` 가 grid 를 만들거나 build 할 때마다 세대 번호(`_grid_epoch`)를
올리고, 그래프를 쏘기 전에 캡처 당시 세대와 같은지 확인한다. 다르면 다시 캡처한다.

### 🔴 그래프는 그래디언트 경로에서 기본으로 끈다 (`--graph_in_grad False`)

**같은 시작 상태에서 20 스텝을 돌렸을 때, eager 실행은 완벽히 재현되지만
(새 Rollout, 재실행 모두 `max|diff| = 0`) 그래프 실행은 eager 와 부동소수점
수준으로 갈린다.** 유동이 발달한 뒤부터만 나타난다.

```
   step0=  0..180   graph vs eager  max|dx| 0          max|dv| 0
   step0=200        graph vs eager  max|dx| 3.7e-09    max|dv| 6.6e-07
   step0=320        graph vs eager  max|dx| 1.3e-06    max|dv| 2.2e-04
   (같은 구간에서 eager vs eager 는 전부 정확히 0)
```

물리적으로는 둘 다 똑같이 타당한 해다. 문제는 **checkpoint 를 만든 전진과 테이프
재현이 정확히 같아야 한다**는 점이다. 1단계 전진만 그래프로 돌리면 재현 구간이
미세하게 다른 궤적을 밟고, T=400 의 adjoint 가 그 차이를 약 `1e4` 배로 증폭시킨다.

```
  graph_in_grad=True   ->  ckpt-vs-full  1.2e-03   (T=400)
  graph_in_grad=False  ->  ckpt-vs-full  5.6e-07   (T=400)
```

그래서 **loss 전진과 checkpoint 1단계 전진은 항상 eager 로 돌린다.**
순방향 시뮬레이션(`--mode forward`, 목표 상태 생성)만 그래프를 탄다.
`--graph_in_grad True` 로 켤 수는 있고, 위 대가를 알고 쓰면 된다.

## 7. 자동미분

`wp.Tape` 는 커널 런치를 기록한다. 다중 스텝을 역전파하려면 두 가지를 지켜야 한다.

1. **스텝마다 상태 배열을 따로 잡는다.** in-place 갱신 금지.
   `vel_pos_step` 이 `pos_in → pos_out` 으로 out-of-place 인 이유다.
2. **스텝마다 HashGrid 를 따로 잡는다.** grid 객체 하나를 `build()` 로 덮어쓰면
   **정방향 결과는 완전히 같고 역방향만 조용히 틀린다.** 앞 스텝의 backward 가
   마지막 스텝의 이웃 목록으로 미분하기 때문이다.
   `checkpoint.backward_taped()` 가 `grids = [new_grid(...) for _ in range(n)]` 로
   리스트에 붙잡아 두는 이유다 (GC 되면 use-after-free).

`grid.build()` 는 미분 대상이 아니므로 항상 `with tape:` **밖**에서 부른다.

## 8. Recursive checkpointing

전 구간을 한 테이프에 올리면 상태 메모리가 `O(T)` 로 자란다.
구간을 나눠 경계 상태만 저장하고 역방향에서 구간을 다시 기록하면 줄일 수 있다.
깊이 `r` 로 재귀하면

```
메모리  ~ (r+1) · T^(1/(r+1)) · S          정방향 계산량 ~ (r+1) · T
```

`checkpoint.backward_rollout()` 이 이것을 구현한다. 깊이 `depth` 에서
세그먼트 길이를 `L = ⌈T^(depth/(depth+1))⌉` 로 잡아 세그먼트 개수가
`T^(1/(depth+1))` 이 되게 하고, 각 세그먼트를 `depth-1` 로 재귀한다.
`--ckpt_depth 0` 이면 전 구간을 한 테이프에 올리는 기준 구현이 된다.

경계에서 adjoint 를 갈아끼우는 것은 `tape.backward(grads={arr: g})` 다.
이것은 누적이 아니라 **대입**이라 세그먼트를 이어붙일 수 있다.
넘겨받은 adjoint 는 반드시 **별도 배열로 복사**한다 — `grad` 는 다음 세그먼트에서
0 으로 지워진다.

저장하는 상태는 `(pos, vel)` 뿐이다. `h` 는 실행 전에 정해지는 입력이라
스텝 사이에 변하지 않는다.

### 실측 (입자 1252개, RTX 3090)

| T | depth | peak GPU mem | 시간 | dL/d(offset) |
|--:|--:|--:|--:|--|
| 400 | 0 | 117.1 MiB | 0.55 s | (-0.132118, -0.029645) |
| 400 | 1 | 7.5 MiB | 0.54 s | (-0.132118, -0.029645) |
| 400 | **2** | **3.7 MiB** | 0.65 s | (-0.132118, -0.029645) |
| 400 | 3 | 3.7 MiB | 0.72 s | (-0.132118, -0.029645) |
| 1600 | 0 | 467.8 MiB | 2.08 s | (-12840.01, -159748.06) |
| 1600 | 1 | 14.5 MiB | 2.19 s | (-12840.37, -159748.11) |
| 1600 | **2** | **5.4 MiB** | 2.55 s | (-12840.10, -159748.34) |
| 1600 | 3 | 4.1 MiB | 2.91 s | (-12839.58, -159747.84) |

`depth=0` 의 메모리는 `T` 에 비례하고(117 → 468 MiB), `depth=2` 는 거의 평평하다
(3.7 → 5.4 MiB). `T=1600` 기준 **87배 절감에 시간은 1.23배**다.

## 9. 검증 — `python main.py --mode check`

Warp 는 틀린 그래디언트를 에러 없이 돌려주므로 **실제 dt 로** 유한차분과 대조한다.
`check` 는 먼저 궤적 발산(증폭률)을 재서 미분이 쓸모 있는 구간을 확인하고,
그 구간에서만 checkpointing 일치를 판정한다.

**GPU, cubic spline, 고정 h (입자 1252개)**

```
     T     증폭   |               AD full |               AD ckpt |                    FD | ckpt-full     AD-FD | 판정
   100    0.99   | ( -0.16000, -0.08000) | ( -0.16000, -0.08000) | ( -0.16000, -0.08002) |  9.31e-07  1.09e-04 | PASS
   200    0.99   | ( -0.16000, -0.08000) | ( -0.16000, -0.08000) | ( -0.16002, -0.08002) |  1.86e-07  1.43e-04 | PASS
   400    0.97   | ( -0.13213, -0.02970) | ( -0.13213, -0.02970) | ( -0.12866, -0.01856) |  3.38e-07  8.43e-02 | PASS
   800   61.3    | ( -0.22339, +0.29517) | ( -0.22339, +0.29517) | ( -0.06547, -0.01591) |  4.75e-06  1.05e+00 | 카오스 구간
```

**CPU, 커널 × h 네 조합 (입자 471개, dx=0.04)** — `ckpt-full` 은 전부 정확히 0,
자동미분이 유한차분과 5% 안에서 맞는 구간도 네 경우 모두 T = 200 이었다.

| 커널 | h | support | 판정 |
|---|--:|--:|---|
| cubic | 0.0520 (auto = 1.3dx) | 0.1040 | PASS |
| cubic | 0.0800 (given) | 0.1600 | PASS |
| wendland | 0.0520 (auto = 1.3dx) | 0.1040 | PASS |
| wendland | 0.0800 (given) | 0.1600 | PASS |

CPU 에서 `ckpt-full` 이 정확히 0 인 것은 atomic 누적 순서가 결정적이기 때문이다.
GPU 에서는 그 순서가 실행마다 달라져 `1e-7` 수준의 차이가 남는다.

커널 단위로도 확인했다 (별도 스크립트, 상대오차):

| 대상 | 결과 |
|---|---|
| `∂rho/∂pos` | 2.6e-4 로 유한차분과 일치 |
| `∂acc/∂vel` (점성) | 1.0e-5 로 일치 |
| `∂acc/∂pos` (압력+점성), 압축된 상태 | 2.7e-5 로 일치 |
| `∂acc/∂pos`, **정지 초기 상태** | 불일치 — 아래 9.2 참고 |

### 9.1 미분 가능한 구간 — 이 문제의 진짜 한계

| T | 증폭 | AD | FD |
|--:|--:|--|--|
| 200 | 0.99 | (-0.160, -0.080) | (-0.160, -0.080) |
| 300 | 1.00 | (-0.158, -0.050) | (-0.158, -0.048) |
| 400 | 0.98 | (-0.132, -0.030) | (-0.129, -0.019) |
| 500 | 1.32 | (-0.123, -0.091) | (-0.076, +0.011) |
| 600 | 4.57 | (-0.170, -0.166) | (-0.062, +0.005) |
| 800 | 69.4 | (-0.220, +0.355) | (-0.066, -0.016) |

**유체 블록이 바닥에 닿아 옆으로 퍼지기 시작하는 순간(T≈450) 부터 궤적이
지수적으로 발산한다.** 그때부터 자동미분은 여전히 그 지점의 정확한 국소
도함수를 주지만, 유한한 eps 의 유한차분과는 갈라지고 하강 방향으로도 쓸모가
없어진다.

이것은 구현 결함이 아니라 dam break 라는 문제 자체의 성질이다.

### 9.2 만나서 정리해 둔 함정 다섯 가지

1. **`offset.grad` 누적.** `Tape.backward` 는 `.grad` 에 **누적**한다.
   최적화 반복 사이에 재사용하는 배열은 반드시 `grad.zero_()` 해야 한다.
   지우지 않으면 에러 없이 매 반복의 그래디언트가 더해져 최적화가 발산한다.

2. **압력 clamp 의 kink.** `--clamp_negative_pressure True` 는 `wp.max(p, 0)` 라
   `rho = rho0` 에 꺾임이 있다. 격자 위에 정지해 있는 **초기 상태는 모든 유체
   입자가 정확히 그 꺾임 위에 앉아 있어서**, 그 지점의 커널 단위 유한차분이
   AD 와 크게 다르게 나온다. clamp 를 끄면 같은 지점에서 AD 와 FD 가 9e-5 로
   일치하고, 100 스텝 굴려 압축된 상태에서는 clamp 를 켠 채로도 2.7e-5 로
   일치한다. 즉 AD 는 자기가 서 있는 조각에 대해 정확하다.
   다만 **clamp 를 끄면 인장 불안정으로 그래디언트가 `1e13` 까지 폭발**하므로
   기본값은 켜 두는 쪽이 맞다.

3. **float32 유한차분.** `sum(rho²) ~ 1e9` 같은 큰 손실에 `eps=1e-6` 을 쓰면
   차이가 float32 해상도 아래로 내려가 FD 가 0 이나 NaN 이 된다.
   `wp.autograd.gradcheck` 도 같은 이유로 FAIL 을 뱉는다.

4. **CUDA graph 와 HashGrid — 무효화.** 6절 참고. 캡처 후 다른 HashGrid 를 쓰면
   그래프가 조용히 깨진다. 세대 번호로 방어했다.

5. **CUDA graph 와 HashGrid — 재현성.** 6절 참고. 그래프는 eager 와 부동소수점
   수준으로 갈린다. 그래디언트 경로에서는 기본으로 끈다.

## 10. 최적화 — 초기 블록 위치 복원

목표 상태는 참값 offset `(0.10, 0.05)` 으로 400 스텝 돌린 최종 위치다.
`(0.02, 0.01)` 에서 출발해

```
초기화(offset → pos0) → forward simulation → loss → backward → 위치 갱신
```

을 반복한다. 손실은 유체 입자의 최종 위치 평균제곱오차다.

**GPU, 기본 설정 (입자 1252개, 400 스텝 × 60 반복)**

```
iter   0  loss 7.348911e-03  offset (+0.02000, +0.01000)  |offset-true| 0.08944
iter  19  loss 2.184966e-04  offset (+0.11412, +0.04562)  |offset-true| 0.01478
iter  39  loss 2.749398e-05  offset (+0.10507, +0.05135)  |offset-true| 0.00524
iter  59  loss 1.270671e-06  offset (+0.10107, +0.05036)  |offset-true| 0.00113
복원된 offset (+0.10101, +0.05031)   참값 (+0.10000, +0.05000)   오차 0.00105
```

**손실 5800배 감소, offset 오차 0.00105 m = 입자 간격의 0.05배.** 39초.

그래디언트 경로는 세 단계다.

```
offset --(tape0)--> pos0 --(recursive checkpoint/replay)--> pos_T --(해석적)--> L
```

`dL/dpos_T` 는 손실이 제곱합이라 해석적으로 바로 쓴다(`loss_seed_cal`).
그래서 손실값을 얻기 위한 순방향 한 번이 추가로 든다 — 총 정방향 계산량은
`(r+2)T` 다. 대신 checkpointing 재귀에 손실 처리가 섞이지 않아 코드가 단순해진다.

옵티마이저는 `warp.optim.Adam` 이다. Adam 의 한 스텝 이동량은 그래디언트 크기와
거의 무관하게 `lr` 이라, `--opt_lr` 은 **입자 간격보다 충분히 작게** 잡아야 한다
(기본 0.01 = dx 의 절반). 수렴 후 진동을 막으려고 `--opt_lr_decay` 로 감쇠시킨다.
`--opt_lr 0.02`(= dx) 로 두면 유체 블록이 벽 입자 위로 올라타 물리가 깨진다.
그래서 offset 은 수조 안으로 clamp 한다(`optimize.clamp_offset`).

> 시작점을 `(0, 0)` 으로 두지 않은 이유: 그 자리에서는 유체 블록이 바닥에
> 정확히 얹혀 있어 목표(2.5dx 떠 있다가 떨어지는 상태)와 동역학 자체가 다르고,
> 그래디언트가 `(+0.030, +0.102)` 로 목표의 반대쪽을 가리킨다.
> 유한차분도 같은 부호(`(+0.054, +0.131)`)를 주므로 구현 문제가 아니라
> 그 지점이 목표의 basin 밖이라는 뜻이다.

## 11. 출력

| 파일 | 내용 |
|---|---|
| `outputs/forward.gif` | 순방향 dam break. 유체는 속도 크기로 색을 준다 |
| `outputs/optimize.gif` | 왼쪽은 현재/목표 초기 블록, 오른쪽은 현재/목표 최종 상태 |
| `outputs/loss_curve.png` | 손실 곡선 |
| `outputs/simulation_trajectory.npy` | `[time, #ptl, 5(x, y, vx, vy, m)]` |
| `outputs/boundary.npy` | `[#bnd, 5(x, y, vx, vy, m)]` |

뒤의 두 파일은 **Reference 코드와 같은 포맷**이라 기존 도구가 그대로 읽는다.

```python
import numpy as np
from visualize.animate import animate            # Reference/visualize/animate.py

traj = np.load("outputs/simulation_trajectory.npy")
bnd = np.load("outputs/boundary.npy")
# 궤적은 이미 frame_step 간격으로 뽑혀 있으므로 프레임 간격을 dt 로 넘긴다.
m = animate(data=traj, bnd=bnd, x=2.0, y=1.0, t=1.0, dt=40 * 2.075e-4)
m.animation_create()
```

기본 설정(`frame_step · dt = 0.0083`)으로 실제 동작을 확인했다.
주의할 점 두 가지: `animation_create()` 의 프레임 stride 가 `int(1/(dt*100))` 이라
넘기는 `dt` 가 `0.01` 보다 크면 stride 가 0 이 되어 `range()` 에서 죽는다.
그리고 writer 가 `imagemagick` 으로 고정돼 있는데, 설치돼 있지 않으면 matplotlib
가 Pillow 로 대체하고 경고만 낸다. `--save_npy False` 로 저장을 끌 수 있다.

## 12. 성능

| dx | 입자 | graph off | graph on | 배수 |
|---|--:|--:|--:|--:|
| 0.02 | 1,252 | 5,268 steps/s | **7,900 steps/s** | **1.50x** |
| 0.008 | 5,371 | 4,782 steps/s | 5,866 steps/s | 1.23x |

입자가 많아지거나 `h/dx` 를 키워 커널 자체가 무거워질수록 런치 오버헤드의
비중이 줄어 이득이 작아진다.

> **주의** — 12절의 성능 수치와 8절의 메모리 표는 이번 구조 변경 **전** 코드에서
> 잰 값이다. 물리와 스텝당 커널 수는 그대로지만 `kernel_w` / `kernel_dwdr` 에
> 커널 종류 분기가 하나 늘었고 grid 세대 확인이 추가됐다. 구조 변경 후에는
> GPU 장애로 재측정하지 못했다 (아래 참고). 재측정이 필요하다.

### GPU 장애 기록

작업 중 `GPU 0 (0000:B4:00.0)` 이 버스에서 떨어졌다.
`nvidia-smi -L` 이 `Unable to determine the device handle` 을 내고, CUDA 드라이버
초기화 자체가 실패해서 `CUDA_VISIBLE_DEVICES` 로 다른 GPU 를 지정해도 Warp 가
`cpu` 만 인식한다. GPU 리셋(`nvidia-smi -r -i 0`) 이나 재부팅이 필요하다.

그래서 구조 변경 후 검증 중 **9절의 CPU 표(커널 × h 네 조합)와 10절의 CPU
최적화**는 새로 돌렸고, GPU 쪽은 장애 직전까지 확인한 것만 남겼다.
GPU 로 재확인이 남은 항목:

- 12절 성능 재측정, 8절 메모리 표 재측정
- Wendland 커널과 h 값 변경 조합의 GPU 검증 (CPU 로는 PASS)
- 6절 그래프 무효화의 방아쇠 분리 (생성 / build / eager 실행 중 무엇인지)

## 13. 주요 설정값

| 플래그 | 기본값 | 뜻 |
|---|---|---|
| `--device` | `cuda:0` | 사용할 디바이스 |
| `--dx` | 0.02 | 입자 간격. `h = 1.3 dx`, `mass = rho0 dx²` |
| `--tank_width` / `--tank_height` | 2.0 / 1.0 | 수조 내부 크기 |
| `--fluid_width` / `--fluid_height` | 0.5 / 0.5 | 초기 유체 블록 |
| `--bnd_layer` | 3 | dummy boundary 겹 수 |
| `--rho0` / `--gamma` / `--c0` | 1000 / 7 / auto | Tait 파라미터 |
| `--mu` | 0.05 | 점성계수 |
| `--kernel_type` | `cubic` | `cubic` 또는 `wendland` |
| `--clamp_negative_pressure` | True | 음압을 0 으로 자른다 |
| `--h` | 0 (auto) | smoothing length. 0 이면 `h_factor · dx` |
| `--h_factor` | 1.3 | `--h` 가 0 일 때 h 를 정하는 비율 |
| `--shepard` / `--shepard_step` | True / 20 | Shepard filter 사용 여부와 주기 |
| `--dt` / `--cfl` | auto / 0.25 | `dt = cfl · h / c0` |
| `--n_steps` / `--frame_step` | 4000 / 40 | 순방향 길이와 프레임 간격 |
| `--grid_dim` | 128 | HashGrid 해시 버킷 한 변 |
| `--use_cuda_graph` | True | 테이프 없는 전진을 CUDA graph 로 묶는다 |
| `--graph_min_steps` | 20 | 그래프 하나가 담을 최소 스텝 수 |
| `--graph_in_grad` | False | 그래디언트 경로에서도 그래프를 쓸지 (6절) |
| `--ckpt_depth` | 2 | recursive checkpointing 깊이 r |
| `--ckpt_min_segment` | 8 | 이보다 짧은 구간은 통째로 테이프에 올린다 |
| `--opt_steps` / `--opt_sim_steps` | 60 / 400 | 최적화 반복 / 시뮬레이션 길이 |
| `--opt_lr` / `--opt_lr_decay` | 0.01 / 0.95 | Adam 학습률과 감쇠 |
| `--true_offset_x/y` | 0.10 / 0.05 | 목표를 만든 참 offset |
| `--init_offset_x/y` | 0.02 / 0.01 | 최적화 시작점 |
| `--fps` / `--dpi` | 20 / 80 | 애니메이션 설정 |
| `--save_npy` | True | 궤적을 Reference 포맷 .npy 로 저장 |

## 14. 아직 없는 것

- CFL 기반 adaptive timestep
- δ-SPH diffusion, XSPH, tensile instability 보정
- 입자마다 다른 h (지금 h 는 전체가 같은 스칼라 입력이다) 와 grad-h(`Ω_i`) 보정항
- 가변 질량
- 다중 GPU
- 테이프 구간의 CUDA graph 캡처 (6절의 이유로 열어 두지 않았다)
