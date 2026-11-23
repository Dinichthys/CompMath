"""Lab 1, stage 8.1: assemble and inspect the R1 finite-difference system.

Grid sizes Nx, Ny count nodes including the boundary. Unknowns are interior
nodes, numbered m = (i - 1) + (j - 1) * (Nx - 2), with x changing fastest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm


def a(x: float, y: float) -> float:
    return 1.0 + 0.3 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def c(x: float, y: float) -> float:
    return 1.0


def f(x: float, y: float) -> float:
    return 10.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)


def g(x: float, y: float) -> float:
    return 0.0


def build_system(Nx: int, Ny: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Discretize -div(a grad u) + c*u = f on [0,1]^2, u|boundary=g."""
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx and Ny must each be at least 3")

    hx, hy = 1.0 / (Nx - 1), 1.0 / (Ny - 1)
    n = (Nx - 2) * (Ny - 2)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    b = np.empty(n, dtype=float)

    def index(i: int, j: int) -> int:
        return (i - 1) + (j - 1) * (Nx - 2)

    for j in range(1, Ny - 1):
        y = j * hy
        for i in range(1, Nx - 1):
            x = i * hx
            m = index(i, j)
            ac = a(x, y)
            neighbors = (
                (i - 1, j, (ac + a(x - hx, y)) / (2.0 * hx * hx)),
                (i + 1, j, (ac + a(x + hx, y)) / (2.0 * hx * hx)),
                (i, j - 1, (ac + a(x, y - hy)) / (2.0 * hy * hy)),
                (i, j + 1, (ac + a(x, y + hy)) / (2.0 * hy * hy)),
            )
            diagonal = c(x, y)
            b[m] = f(x, y)
            for ni, nj, weight in neighbors:
                diagonal += weight
                if 1 <= ni <= Nx - 2 and 1 <= nj <= Ny - 2:
                    rows.append(m)
                    cols.append(index(ni, nj))
                    values.append(-weight)
                else:
                    b[m] += weight * g(ni * hx, nj * hy)
            rows.append(m)
            cols.append(m)
            values.append(diagonal)

    A = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    A.sort_indices()
    return A, b


def inspect_system(Nx: int = 8, Ny: int = 8, output: Path | None = None) -> str:
    A, b = build_system(Nx, Ny)
    n = A.shape[0]
    relative_asymmetry = sparse_norm(A - A.T) / sparse_norm(A)
    if Nx == Ny == 8:
        np.linalg.cholesky(A.toarray())  # Small-grid SPD check only.
        cholesky = "успешно"
    else:
        cholesky = "пропущено (проверяется только для сетки 8×8)"

    tests = {
        "постоянный": np.ones(n),
        "знакопеременный": np.array([(-1.0) ** (i + j) for j in range(1, Ny - 1) for i in range(1, Nx - 1)]),
        "синусоидальный": np.array([np.sin(np.pi * i / (Nx - 1)) * np.sin(np.pi * j / (Ny - 1)) for j in range(1, Ny - 1) for i in range(1, Nx - 1)]),
    }
    lines = [
        "Лабораторная работа №1, этап 8.1, вариант №11 = R1",
        f"Сетка: Nx={Nx}, Ny={Ny}; внутренних неизвестных: {n}",
        f"Матрица: {n}×{n}; ненулевых элементов: {A.nnz}",
        f"Максимум ненулевых элементов в строке: {np.diff(A.indptr).max()}",
        f"Относительная асимметрия: {relative_asymmetry:.3e}",
        f"Машинный эпсилон float64: {np.finfo(np.float64).eps:.3e}",
        f"Минимальный диагональный элемент: {A.diagonal().min():.6f}",
        f"Разложение Холецкого: {cholesky}",
    ]
    for name, z in tests.items():
        lines.append(f"zᵀAz ({name}): {z @ (A @ z):.6f}")
    if Nx == Ny == 8:
        m = (3 - 1) + (3 - 1) * (Nx - 2)
        row = A.getrow(m)
        terms = ", ".join(f"A[{m},{col}]={val:.6f}" for col, val in zip(row.indices, row.data))
        equation = " ".join(f"{val:+.6f}u_{col}" for col, val in zip(row.indices, row.data)).lstrip("+")
        lines.extend([
            f"Строка для узла (i,j)=(3,3), m={m}: {terms}",
            f"b[{m}]={b[m]:.6f}",
            f"Уравнение строки m={m}: {equation} = {b[m]:.6f}",
        ])
    if output is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.spy(A, markersize=max(0.25, min(4.0, 120.0 / max(Nx, Ny))))
        ax.set_title(f"Портрет матрицы A, {Nx}×{Ny}")
        ax.set_xlabel("столбец")
        ax.set_ylabel("строка")
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        lines.append(f"Портрет матрицы: {output}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--plot", type=Path, default=Path(__file__).with_name("lab1_stage_8_1_R1_spy.png"))
    args = parser.parse_args()
    print(inspect_system(args.nx, args.ny, args.plot))


if __name__ == "__main__":
    main()
