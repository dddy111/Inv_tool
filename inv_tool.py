# -*- coding: utf-8 -*-
from fractions import Fraction
import random
import time

# ============================================================
# 기본 유틸리티 함수 모음
# - 행렬의 크기, 복사, 출력, 곱셈, 오차 계산 등의 공통 기능
# ============================================================

def mat_shape(A):
    """
    행렬 A의 (행, 열) 크기를 튜플 형태로 반환한다.
    A가 비어 있으면 (0, 0)을 반환한다.
    """
    return (len(A), len(A[0]) if A else 0)


def is_square(A):
    """
    행렬 A가 정방행렬인지 여부를 반환한다.
    즉, 행의 수와 열의 수가 같고 0보다 커야 한다.
    """
    n, m = mat_shape(A)
    return (n == m) and (n > 0)


def deepcopy_mat(A):
    """
    얕은 복사가 아닌, 행 단위로 완전한 복사본을 생성한다.
    원본 A의 변경이 복사본에 영향을 주지 않도록 한다.
    """
    return [row[:] for row in A]


def eye(n):
    """
    n×n 크기의 단위행렬(Identity matrix)을 Fraction 형태로 생성한다.
    """
    I = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(Fraction(1, 1) if i == j else Fraction(0, 1))
        I.append(row)
    return I


def pretty_entry(x: Fraction):
    """
    Fraction 객체를 사람이 보기 쉽게 문자열로 변환한다.
    분모가 1이면 정수로, 그렇지 않으면 'p/q' 형태로 출력한다.
    """
    return f"{x.numerator}" if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def print_matrix(A, title=None, width=8):
    """
    행렬을 고정폭으로 정렬하여 콘솔에 출력한다.
    Fraction 타입의 원소를 깔끔하게 맞춰서 보여준다.
    """
    if title:
        print(title)
    if not A:
        print("[]\n")
        return

    as_text = [[pretty_entry(v) for v in row] for row in A]
    colw = []
    for c in range(len(as_text[0])):
        colw.append(max(len(as_text[r][c]) for r in range(len(as_text))))
        colw[-1] = max(colw[-1], width)

    for r in as_text:
        line = " ".join(s.rjust(w) for s, w in zip(r, colw))
        print(line)
    print()


def matmul(A, B):
    """
    행렬 A와 B의 곱을 계산한다.
    차원이 맞지 않으면 예외를 발생시킨다.
    """
    n, k1 = mat_shape(A)
    k2, m = mat_shape(B)
    if k1 != k2:
        raise ValueError("행렬 곱셈 차원 불일치")

    C = [[Fraction(0, 1) for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = Fraction(0, 1)
            for t in range(k1):
                acc += A[i][t] * B[t][j]
            C[i][j] = acc
    return C


def max_abs_diff(A, B):
    """
    두 행렬 A, B의 각 원소 차이의 절댓값 중 최대값을 반환한다.
    """
    if mat_shape(A) != mat_shape(B):
        return None
    n, m = mat_shape(A)
    best = Fraction(0, 1)
    for i in range(n):
        for j in range(m):
            d = A[i][j] - B[i][j]
            if d < 0:
                d = -d
            if d > best:
                best = d
    return best


# ============================================================
# 가우스-조던 소거법 기반 역행렬 계산
# - 부분 피벗팅을 포함하여 수치적으로 안정적인 역행렬 계산
# ============================================================

def gauss_jordan_inverse(A):
    """
    주어진 정방행렬 A의 역행렬을 가우스-조던 소거법으로 계산한다.
    피벗팅을 포함하며, Fraction 연산으로 정확도를 유지한다.
    예외: 특이행렬(역행렬이 존재하지 않는 경우) → ValueError 발생
    """
    if not is_square(A):
        raise ValueError("정방행렬이 아닙니다.")
    n, _ = mat_shape(A)

    M = deepcopy_mat(A)
    Inv = eye(n)

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[pivot][col] == 0:
            raise ValueError("역행렬이 존재하지 않습니다(특이행렬).")

        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            Inv[col], Inv[pivot] = Inv[pivot], Inv[col]

        p = M[col][col]
        invp = Fraction(1, 1) / p
        for j in range(n):
            M[col][j] *= invp
            Inv[col][j] *= invp

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor != 0:
                for j in range(n):
                    M[r][j] -= factor * M[col][j]
                    Inv[r][j] -= factor * Inv[col][j]

    return Inv


# ============================================================
# 행렬식(Determinant)과 여인수행렬(Adjugate) 기반 역행렬 계산
# ============================================================

def minor_matrix(A, i, j):
    """
    행렬 A에서 i행, j열을 제거한 소행렬(minor matrix)을 반환한다.
    """
    n, _ = mat_shape(A)
    return [[A[r][c] for c in range(n) if c != j] for r in range(n) if r != i]


def det_recursive(A):
    """
    여인수 전개를 이용한 재귀적 행렬식 계산.
    소규모 행렬(n≤6)에 적합하다.
    """
    n, _ = mat_shape(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]

    s = Fraction(0, 1)
    sign = 1
    for j in range(n):
        a = A[0][j]
        if a != 0:
            s += sign * a * det_recursive(minor_matrix(A, 0, j))
        sign = -sign
    return s


def adjugate_matrix(A):
    """
    여인수행렬(cofactor matrix)의 전치행렬 adj(A)를 계산한다.
    """
    n, _ = mat_shape(A)
    C = [[Fraction(0, 1) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cof = det_recursive(minor_matrix(A, i, j))
            if (i + j) % 2:
                cof = -cof
            C[i][j] = cof
    return [[C[j][i] for j in range(n)] for i in range(n)]


def inverse_by_adjugate(A, limit=6):
    """
    역행렬 = adj(A) / det(A) 방식으로 계산한다.
    det(A)=0이면 역행렬이 존재하지 않는다.
    n이 너무 크면 (limit 초과) 시간 문제로 제한한다.
    """
    if not is_square(A):
        raise ValueError("정방행렬이 아닙니다.")
    n, _ = mat_shape(A)
    if n > limit:
        raise ValueError(f"행렬식 기반 역행렬은 n ≤ {limit}에서만 허용됩니다.")

    detA = det_recursive(A)
    if detA == 0:
        raise ValueError("역행렬이 존재하지 않습니다(det=0).")

    Adj = adjugate_matrix(A)
    inv = [[Adj[i][j] / detA for j in range(n)] for i in range(n)]
    return inv, detA


# ============================================================
# 난수 행렬 생성, 정확도 평가, 시간 측정 (그래프가 의존)
# ============================================================

def gen_random_invertible_matrix(n, lo=-5, hi=5, trials=500):
    """
    주어진 범위에서 난수로 정방행렬을 생성하고,
    가우스-조던 역행렬이 존재하는 경우에만 반환한다.
    """
    for _ in range(trials):
        A = [[Fraction(random.randint(lo, hi), 1) for _ in range(n)] for _ in range(n)]
        try:
            _ = gauss_jordan_inverse(A)
            return A
        except Exception:
            pass
    raise RuntimeError("가역 난수 행렬 생성 실패")


def accuracy(A, inv):
    """
    정확도 측정: ||A·inv − I||_max (절대 오차 최대값)
    """
    I = eye(len(A))
    return max_abs_diff(matmul(A, inv), I)


def time_one(A, method="gj", adj_limit=6):
    """
    한 알고리즘(가우스-조던 또는 행렬식 방식)에 대한
    수행 시간, 역행렬, 정확도, 성공 여부를 반환한다.
    """
    t0 = time.perf_counter()
    try:
        if method == "gj":
            inv = gauss_jordan_inverse(A)
        else:
            inv, _ = inverse_by_adjugate(A, limit=adj_limit)
        t1 = time.perf_counter()
        return (t1 - t0), inv, accuracy(A, inv), True
    except Exception:
        return (time.perf_counter() - t0), None, None, False


# ============================================================
# 사용자 입력 / 파싱
# ============================================================

def parse_number(token: str):
    """
    문자열을 Fraction으로 변환한다.
    'a/b' 형태는 분수, 정수는 그대로 처리한다.
    """
    t = token.strip()
    if "/" in t:
        p, q = t.split("/")
        return Fraction(int(p), int(q))
    return Fraction(int(t), 1)


def read_matrix_from_stdin():
    """
    사용자로부터 n×n 행렬을 직접 입력받아 Fraction 형태로 반환한다.
    """
    n = int(input("정방행렬의 크기 n: ").strip())
    if n <= 0:
        raise ValueError("n 은 양의 정수여야 합니다.")
    print(f"{n}×{n} 행렬을 행 단위로 입력(정수 또는 유리수 a/b):")
    A = []
    for _ in range(n):
        row = [parse_number(tok) for tok in input().strip().split()]
        if len(row) != n:
            raise ValueError("열 개수가 맞지 않습니다.")
        A.append(row)
    return A


# ============================================================
# 계산 시간 비교 그래프
# ============================================================

def compare_time_graph(sizes, lo=-5, hi=5, adj_limit=6):
    """
    행렬 크기 n에 따른 두 방식(Gauss-Jordan, Adjugate)의
    계산 시간을 matplotlib 그래프로 비교하여 시각화한다.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib가 설치되어 있지 않습니다. (설치: pip install matplotlib)")
        return

    gj_x, gj_y = [], []
    adj_x, adj_y = [], []

    print("=== 계산 시간 비교 ===")
    for n in sizes:
        print(f"- n={n}")
        A = gen_random_invertible_matrix(n, lo, hi)

        t_gj, _, _, ok_gj = time_one(A, "gj", adj_limit)
        if ok_gj:
            gj_x.append(n); gj_y.append(t_gj)

        t_adj, _, _, ok_adj = time_one(A, "adj", adj_limit)
        if ok_adj:
            adj_x.append(n); adj_y.append(t_adj)

    plt.figure()
    if gj_x:
        plt.plot(gj_x, gj_y, "o-", label="Gauss-Jordan")
    if adj_x:
        plt.plot(adj_x, adj_y, "o-", label="Adjugate")
    plt.xlabel("Matrix size n")
    plt.ylabel("Time (s)")
    plt.title("Inverse Computation Time (Fraction arithmetic)")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# 메인 실행부
# ============================================================

def main():
    """
    프로그램의 진입점.
    메뉴를 통해 수동 입력(필수 기능), 그래프 비교(그래프 기능)를 제공한다.
    """
    random.seed(0xC0FFEE)

    print("메뉴:")
    print("  1) 수동 입력 → 두 방법으로 역행렬 계산·출력·비교")
    print("  2) 계산 시간 비교 그래프")
    choice = input("선택: ").strip()

    if choice == "1":
        A = read_matrix_from_stdin()
        print_matrix(A, "입력 행렬 A")

        # 가우스-조던 방식 계산
        t0 = time.perf_counter()
        try:
            inv_gj = gauss_jordan_inverse(A)
            t1 = time.perf_counter()
            print_matrix(inv_gj, f"[가우스-조던 역행렬] 소요 {t1 - t0:.6f}s")
        except Exception as e:
            t1 = time.perf_counter()
            print(f"가우스-조던 실패: {e} (소요 {t1 - t0:.6f}s)")
            inv_gj = None

        # 행렬식(adjugate) 방식 계산
        t2 = time.perf_counter()
        try:
            inv_adj, detA = inverse_by_adjugate(A, limit=6)
            t3 = time.perf_counter()
            print(f"det(A) = {pretty_entry(detA)}")
            print_matrix(inv_adj, f"[행렬식(adjugate) 역행렬] 소요 {t3 - t2:.6f}s")
        except Exception as e:
            t3 = time.perf_counter()
            print(f"행렬식(adjugate) 실패: {e} (소요 {t3 - t2:.6f}s)")
            inv_adj = None

        # 두 결과 비교
        if (inv_gj is not None) and (inv_adj is not None):
            I = eye(len(A))
            diff = max_abs_diff(inv_gj, inv_adj)
            d1 = max_abs_diff(matmul(A, inv_gj), I)
            d2 = max_abs_diff(matmul(A, inv_adj), I)
            print(f"두 방법 최대 절대 차이: {pretty_entry(diff)}")
            print(f"검증 ||A*inv_gj - I||_max = {pretty_entry(d1)}")
            print(f"검증 ||A*inv_adj - I||_max = {pretty_entry(d2)}")
            print("동일성:", "성립" if diff == Fraction(0,1) else "불일치")

    elif choice == "2":
        sizes = list(map(int, input("비교할 n 리스트 (예: 2 3 4 5 6): ").split()))
        lo, hi = map(int, input("난수 범위 [lo hi] (예: -5 5): ").split())
        compare_time_graph(sizes, lo, hi, adj_limit=6)

    else:
        print("잘못된 선택입니다.")


# ============================================================
# 프로그램 시작점
# ============================================================

if __name__ == "__main__":
    main()
