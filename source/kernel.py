"""SPH 커널 함수와 Warp kernel 모음.

배열 표기
    #all = #ptl + #bnd   (유체 입자와 경계 입자를 한 배열에 담는다)
    위치와 속도는 wp.vec3 로 저장하지만 z 성분은 항상 0 이다. 물리는 완전히 2D 이고
    (커널 정규화 상수도 2D 다), vec3 를 쓰는 이유는 Warp 의 HashGrid 가 3D 점만
    받기 때문이다.

한 step 은 아래 순서로 읽힌다.

    이웃 탐색(HashGrid.build)  ->  density_cal  ->  (shepard_filter)
                               ->  pres_cal     ->  force_cal  ->  vel_pos_step

smoothing length h 는 초기 입자 배치와 마찬가지로 실행 전에 정해지는 입력이고
시뮬레이션 동안 바뀌지 않는다. 그래서 커널에는 스칼라로 넘긴다.
"""

import math

import warp as wp

# 입자 종류
PTL = wp.constant(wp.int32(0))          # 유체 입자
BND = wp.constant(wp.int32(1))          # dummy boundary 입자

# 커널 종류 (Config.kernel_id 와 같은 값)
CUBIC = wp.constant(wp.int32(0))
WENDLAND = wp.constant(wp.int32(1))

# r^2 이 이보다 작으면 같은 입자이거나 겹친 입자로 보고 건너뛴다.
# wp.sqrt(0) 의 adjoint 가 inf 가 되는 것을 막는 장치이기도 하다.
R2_MIN = wp.constant(1.0e-12)

# 정규화 상수 중 h 에 의존하지 않는 부분. 전체 상수는 이 값 / h^2 다.
CUBIC_NORM = wp.constant(10.0 / (7.0 * math.pi))        # 2D cubic spline
WENDLAND_NORM = wp.constant(7.0 / (4.0 * math.pi))      # 2D Wendland C2


# ---------------------------------------------------------------- SPH 커널 함수
@wp.func
def kernel_w(r: float, h: float, ker: int) -> float:
    """SPH 커널 W(r, h). 두 커널 모두 지지 반경은 2h 다.

    q = 2 에서 W 와 dW/dr 이 함께 0 이므로 `if r < 2h` 컷오프를 넣어도 자동미분이
    델타 항을 놓치지 않는다. h 로 미분해도 같은 이유로 안전하다 — 가변 h 의 근거다.

    r: 두 입자 사이 거리
    h: 이 쌍에 쓸 smoothing length
    ker: CUBIC 또는 WENDLAND
    """
    q = r / h
    w = float(0.0)
    if ker == CUBIC:
        if q < 1.0:
            w = CUBIC_NORM * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        elif q < 2.0:
            t = 2.0 - q
            w = CUBIC_NORM * 0.25 * t * t * t
    else:
        if q < 2.0:
            u = 1.0 - 0.5 * q
            w = WENDLAND_NORM * u * u * u * u * (1.0 + 2.0 * q)
    return w / (h * h)


@wp.func
def kernel_dwdr(r: float, h: float, ker: int) -> float:
    """SPH 커널의 dW/dr. r < h 구간에서 음수다.

    r: 두 입자 사이 거리
    h: 이 쌍에 쓸 smoothing length
    ker: CUBIC 또는 WENDLAND
    """
    q = r / h
    d = float(0.0)
    if ker == CUBIC:
        if q < 1.0:
            d = CUBIC_NORM * (-3.0 * q + 2.25 * q * q)
        elif q < 2.0:
            t = 2.0 - q
            d = CUBIC_NORM * (-0.75 * t * t)
    else:
        if q < 2.0:
            u = 1.0 - 0.5 * q
            d = WENDLAND_NORM * (-5.0 * q * u * u * u)
    return d / (h * h * h)


# ---------------------------------------------------------------- 초기 배치
@wp.kernel
def place_ptl(
    base_pos: wp.array(dtype=wp.vec3),      # [#all, 3]
    offset: wp.array(dtype=wp.vec3),        # [1, 3]
    ptl_type: wp.array(dtype=wp.int32),     # [#all]
    pos: wp.array(dtype=wp.vec3),           # [#all, 3]  (출력)
) -> None:
    """유체 입자만 offset 만큼 평행이동한다. 최적화의 미분 대상이 offset 이다."""
    i = wp.tid()
    if ptl_type[i] == PTL:
        pos[i] = base_pos[i] + offset[0]
    else:
        pos[i] = base_pos[i]


# ---------------------------------------------------------------- 1. 밀도
@wp.kernel
def density_cal(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),           # [#all, 3]
    h: float,
    mass: float,
    support: float,
    ker: int,
    rho: wp.array(dtype=float),             # [#all]  (출력)
) -> None:
    """rho_i = m * sum_j W(|x_i - x_j|, h). 경계 입자도 합에 들어간다."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)     # 공간 정렬된 slot -> 원래 입자 번호
    pos_i = pos[i]

    s = kernel_w(0.0, h, ker)                # 자기 자신. r 이 항상 0 이라 상수다
    for j in wp.hash_grid_query(grid, pos_i, support):
        r_vec = pos_i - pos[j]
        r2 = wp.dot(r_vec, r_vec)
        if r2 > R2_MIN:
            if r2 < support * support:       # 셀 후보를 실제 지지 반경으로 거른다
                s = s + kernel_w(wp.sqrt(r2), h, ker)

    rho[i] = mass * s


# ---------------------------------------------------------------- 2. 밀도 보정
@wp.kernel
def shepard_filter(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),           # [#all, 3]
    h: float,
    rho_in: wp.array(dtype=float),          # [#all]
    mass: float,
    support: float,
    ker: int,
    rho_out: wp.array(dtype=float),         # [#all]  (출력)
) -> None:
    """rho_i <- sum_j m W_ij / sum_j (m/rho_j) W_ij.

    커널 합의 0차 일관성을 되돌려 주는 필터라 밀도장의 고주파 잡음이 줄어든다.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    pos_i = pos[i]

    w0 = kernel_w(0.0, h, ker)
    num = mass * w0
    den = (mass / rho_in[i]) * w0
    for j in wp.hash_grid_query(grid, pos_i, support):
        r_vec = pos_i - pos[j]
        r2 = wp.dot(r_vec, r_vec)
        if r2 > R2_MIN:
            if r2 < support * support:
                w = kernel_w(wp.sqrt(r2), h, ker)
                num = num + mass * w
                den = den + (mass / rho_in[j]) * w

    rho_out[i] = num / den


# ---------------------------------------------------------------- 3. 압력
@wp.kernel
def pres_cal(
    rho: wp.array(dtype=float),             # [#all]
    rho0: float,
    B: float,
    gamma: float,
    clamp_negative: int,
    pres: wp.array(dtype=float),            # [#all]  (출력)
) -> None:
    """Tait 상태방정식  p = B((rho/rho0)^gamma - 1)."""
    i = wp.tid()
    q = wp.pow(rho[i] / rho0, gamma) - 1.0
    if clamp_negative == 1:
        q = wp.max(q, 0.0)
    pres[i] = B * q


# ---------------------------------------------------------------- 4. 힘
@wp.kernel
def force_cal(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),           # [#all, 3]
    vel: wp.array(dtype=wp.vec3),           # [#all, 3]
    h: float,
    rho: wp.array(dtype=float),             # [#all]
    pres: wp.array(dtype=float),            # [#all]
    mass: float,
    support: float,
    ker: int,
    mu: float,
    g: float,
    acc: wp.array(dtype=wp.vec3),           # [#all, 3]  (출력)
) -> None:
    """압력력(대칭형) + 점성력(Morris) + 중력.

    셋을 커널 하나에 묶은 이유는 이웃 순회를 한 번만 돌기 위해서다. 조밀행렬
    구현이라면 나눠도 공짜지만, HashGrid 에서는 나눌 때마다 탐색이 한 벌씩 는다.

    두 항 모두 이웃 루프에서 순수 누적(`+=`)만 하므로 동적 루프에서도 adjoint 가 정확하다.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    pos_i = pos[i]
    vel_i = vel[i]
    rho_i = rho[i]
    pres_i = pres[i]

    a = wp.vec3(0.0, 0.0, 0.0)
    for j in wp.hash_grid_query(grid, pos_i, support):
        r_vec = pos_i - pos[j]
        r2 = wp.dot(r_vec, r_vec)
        if r2 > R2_MIN:
            if r2 < support * support:
                r = wp.sqrt(r2)
                grad_w = kernel_dwdr(r, h, ker) * (r_vec / r)        # grad_i W_ij
                rho_j = rho[j]

                # 압력력: -m (p_i/rho_i^2 + p_j/rho_j^2) grad W
                a = a - mass * (pres_i / (rho_i * rho_i)
                                + pres[j] / (rho_j * rho_j)) * grad_w

                # 점성력: Morris 형. dot(r_vec, grad_w) < 0 이라 속도 차를 감쇠시킨다
                visc = 2.0 * mu * wp.dot(r_vec, grad_w) \
                    / (rho_i * rho_j * (r2 + 0.01 * h * h))
                a = a + mass * visc * (vel_i - vel[j])

    acc[i] = a + wp.vec3(0.0, -g, 0.0)


# ---------------------------------------------------------------- 5. 적분
@wp.kernel
def vel_pos_step(
    pos_in: wp.array(dtype=wp.vec3),        # [#all, 3]
    vel_in: wp.array(dtype=wp.vec3),        # [#all, 3]
    acc: wp.array(dtype=wp.vec3),           # [#all, 3]
    ptl_type: wp.array(dtype=wp.int32),     # [#all]
    dt: float,
    pos_out: wp.array(dtype=wp.vec3),       # [#all, 3]  (출력)
    vel_out: wp.array(dtype=wp.vec3),       # [#all, 3]  (출력)
) -> None:
    """semi-implicit Euler. 경계 입자는 움직이지 않는다.

    속도와 위치를 커널 하나에서 갱신하는 이유는 위치가 갱신된 속도를 써야 하기
    때문이다. 입력 배열과 출력 배열을 따로 두어 in-place 갱신을 피한다 (자동미분 요건).
    """
    i = wp.tid()
    if ptl_type[i] == PTL:
        v_new = vel_in[i] + acc[i] * dt
        vel_out[i] = v_new
        pos_out[i] = pos_in[i] + v_new * dt
    else:
        vel_out[i] = vel_in[i]
        pos_out[i] = pos_in[i]


# ---------------------------------------------------------------- 손실
@wp.kernel
def loss_cal(
    pos: wp.array(dtype=wp.vec3),           # [#all, 3]
    pos_target: wp.array(dtype=wp.vec3),    # [#all, 3]
    ptl_type: wp.array(dtype=wp.int32),     # [#all]
    inv_n: float,
    loss: wp.array(dtype=float),            # [1]  (출력)
) -> None:
    """L = (1/n_ptl) * sum_ptl |x_i - x_target_i|^2"""
    i = wp.tid()
    if ptl_type[i] == PTL:
        d = pos[i] - pos_target[i]
        wp.atomic_add(loss, 0, inv_n * wp.dot(d, d))


@wp.kernel
def loss_seed_cal(
    pos: wp.array(dtype=wp.vec3),           # [#all, 3]
    pos_target: wp.array(dtype=wp.vec3),    # [#all, 3]
    ptl_type: wp.array(dtype=wp.int32),     # [#all]
    inv_n: float,
    seed: wp.array(dtype=wp.vec3),          # [#all, 3]  (출력)
) -> None:
    """dL/dx_T 를 해석적으로 채운다. 역전파의 씨앗이다."""
    i = wp.tid()
    if ptl_type[i] == PTL:
        seed[i] = 2.0 * inv_n * (pos[i] - pos_target[i])
    else:
        seed[i] = wp.vec3(0.0, 0.0, 0.0)
