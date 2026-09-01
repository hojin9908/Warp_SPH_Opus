"""SOPHIA 형식의 입자 입력 파일을 읽고 쓴다.

파일 형식 (SOPHIA `input/input.txt` 와 같다)

    1행     컬럼 ID 를 탭으로 나열한 헤더
    2행~    입자 하나씩, 헤더와 같은 순서로 값을 탭으로 나열

컬럼 ID 는 그 자리에 어떤 물리량이 들어 있는지를 말한다. 그래서 컬럼 순서가
자유롭고, 필요한 것만 넣어도 된다.

    ID  물리량              이 solver 에서
    --  ------------------  ----------------------------------------
     1  x                   입자별로 읽는다
     2  y                   입자별로 읽는다
     3  z                   읽되 0 이 아니면 거부한다 (2D solver 다)
     4  ux                  입자별로 읽는다
     5  uy                  입자별로 읽는다
     6  uz                  읽되 0 이 아니면 거부한다
     7  m (질량)            읽어서 설정을 덮어쓴다. 입자마다 다르면 거부한다
     8  p_type              0 = 경계, 1 = 유체  (SOPHIA 규약)
     9  h                   읽어서 설정을 덮어쓴다. 입자마다 다르면 거부한다
    그 외                   무시하고 한 번 알려 준다

질량과 h 를 "입자마다 다르면 거부" 하는 이유는 이 solver 가 둘 다 전역 스칼라로
쓰기 때문이다. 조용히 무시하는 대신 명시적으로 막는다.
"""

from dataclasses import dataclass

import numpy as np

# SOPHIA 의 컬럼 ID
ID_X, ID_Y, ID_Z = 1, 2, 3
ID_UX, ID_UY, ID_UZ = 4, 5, 6
ID_M = 7
ID_PTYPE = 8
ID_H = 9

ID_NAME = {ID_X: "x", ID_Y: "y", ID_Z: "z", ID_UX: "ux", ID_UY: "uy", ID_UZ: "uz",
           ID_M: "m", ID_PTYPE: "p_type", ID_H: "h"}

# p_type 규약 (SOPHIA 와 같다)
BND = 0     # 경계 입자
PTL = 1     # 유체 입자


@dataclass
class ParticleInput:
    """입력 파일에서 읽어 온 입자들.

    pos: 위치            # [#all, 3]  z 는 0
    vel: 속도            # [#all, 3]  z 는 0
    ptl_type: 입자 종류  # [#all]     PTL 또는 BND
    n_ptl: 유체 입자 수
    mass: 파일이 준 질량. 없으면 None
    h: 파일이 준 smoothing length. 없으면 None
    """
    pos: np.ndarray
    vel: np.ndarray
    ptl_type: np.ndarray
    n_ptl: int
    mass: float | None = None
    h: float | None = None


def _uniform_value(col: np.ndarray, name: str, path: str) -> float:
    """열 전체가 같은 값인지 확인하고 그 값을 돌려준다."""
    lo, hi = float(col.min()), float(col.max())
    if hi - lo > 1e-12 * max(abs(hi), 1.0):
        raise ValueError(
            f"{path}: '{name}' 이 입자마다 다르다 ({lo:g} ~ {hi:g}). "
            f"이 solver 는 {name} 을 전역 스칼라로 쓴다."
        )
    return lo


def read_input(path: str) -> ParticleInput:
    """SOPHIA 형식 입력 파일을 읽는다.

    path: 입력 파일 경로
    return: ParticleInput
    """
    with open(path) as f:
        header = f.readline()
        if not header.strip():
            raise ValueError(f"{path}: 첫 줄이 비어 있다. 컬럼 ID 헤더가 있어야 한다.")
        ids = [int(tok) for tok in header.split()]
        rows = [line.split() for line in f if line.strip()]

    if not rows:
        raise ValueError(f"{path}: 입자가 하나도 없다.")
    bad = [i + 2 for i, r in enumerate(rows) if len(r) != len(ids)]
    if bad:
        raise ValueError(f"{path}: {len(bad)} 개 줄의 값 개수가 헤더({len(ids)}개)와 다르다. "
                         f"처음 몇 줄: {bad[:5]}")

    table = np.array(rows, dtype=np.float64)            # [#all, #col]
    n = len(table)
    col = {cid: table[:, j] for j, cid in enumerate(ids)}

    unknown = sorted(set(ids) - set(ID_NAME))
    if unknown:
        print(f"  [read_input] 이 solver 가 쓰지 않는 컬럼은 무시한다: {unknown}")

    for need in (ID_X, ID_Y, ID_PTYPE):
        if need not in col:
            raise ValueError(f"{path}: 필수 컬럼 {need}({ID_NAME[need]}) 가 없다.")

    # 2D solver 라 z 성분은 0 이어야 한다.
    for cid in (ID_Z, ID_UZ):
        if cid in col and np.abs(col[cid]).max() > 1e-12:
            raise ValueError(f"{path}: '{ID_NAME[cid]}' 가 0 이 아니다. 이 solver 는 2D 다.")

    pos = np.zeros((n, 3), dtype=np.float32)            # [#all, 3]
    pos[:, 0] = col[ID_X]
    pos[:, 1] = col[ID_Y]
    vel = np.zeros((n, 3), dtype=np.float32)            # [#all, 3]
    if ID_UX in col:
        vel[:, 0] = col[ID_UX]
    if ID_UY in col:
        vel[:, 1] = col[ID_UY]

    ptl_type = np.rint(col[ID_PTYPE]).astype(np.int32)  # [#all]
    seen = set(np.unique(ptl_type).tolist())
    if not seen <= {PTL, BND}:
        raise ValueError(f"{path}: 모르는 p_type {sorted(seen - {PTL, BND})}. "
                         f"이 solver 는 {BND}(경계) 와 {PTL}(유체) 만 안다.")

    mass = _uniform_value(col[ID_M], "m", path) if ID_M in col else None
    h = _uniform_value(col[ID_H], "h", path) if ID_H in col else None

    return ParticleInput(pos=pos, vel=vel, ptl_type=ptl_type,
                         n_ptl=int((ptl_type == PTL).sum()), mass=mass, h=h)


def write_input(path: str, pos: np.ndarray, vel: np.ndarray, ptl_type: np.ndarray,
                mass: float, h: float) -> None:
    """생성한 입자를 SOPHIA 형식으로 쓴다. 읽어서 고쳐 쓰기 좋으라고 둔 것이다.

    path: 저장 경로
    pos: 위치            # [#all, 3]
    vel: 속도            # [#all, 3]
    ptl_type: 입자 종류  # [#all]
    mass: 입자 질량
    h: smoothing length
    """
    ids = [ID_X, ID_Y, ID_UX, ID_UY, ID_M, ID_H, ID_PTYPE]
    with open(path, "w") as f:
        f.write("\t".join(str(i) for i in ids) + "\n")
        for i in range(len(pos)):
            f.write("\t".join([
                f"{pos[i, 0]:.6e}", f"{pos[i, 1]:.6e}",
                f"{vel[i, 0]:.6e}", f"{vel[i, 1]:.6e}",
                f"{mass:.6e}", f"{h:.6e}", str(int(ptl_type[i])),
            ]) + "\n")
    print(f"  저장: {path}  ({len(pos)} 입자, 컬럼 ID {ids})")
