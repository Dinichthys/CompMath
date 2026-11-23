"""Lab 1, section 8.2: Jacobi, Gauss-Seidel, and SOR for variant R1.

All runs start from zero and stop at the true normalized residual
eta_s = ||b-Ax_s||_2 / ||b||_2 <= epsilon_res. On the N=32 test system,
u_star is prescribed, so delta_s = ||u_star-x_s||_2 / ||u_star||_2
can also be measured. The main variant has no prescribed exact solution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

from lab1_stage_8_1_R1 import build_system


@dataclass
class Result:
    x: np.ndarray
    iterations: int
    residuals: list[float]
    deltas: list[float] | None
    seconds: float
    converged: bool

    @property
    def eta(self) -> float:
        return self.residuals[-1]

    @property
    def delta(self) -> float | None:
        return None if self.deltas is None else self.deltas[-1]


def solve_stationary(
    A: sparse.csr_matrix,
    b: np.ndarray,
    method: str,
    *,
    tol: float = 1e-8,
    max_iter: int = 100_000,
    omega: float = 1.5,
    u_star: np.ndarray | None = None,
) -> Result:
    """Use the same directly recomputed residual criterion for every method."""
    if method not in {"jacobi", "gauss_seidel", "sor"}:
        raise ValueError("unknown method")
    if not 0 < tol < 1 or max_iter < 1:
        raise ValueError("tol must be in (0,1) and max_iter must be positive")
    if method == "sor" and not 0 < omega < 2:
        raise ValueError("SOR requires 0 < omega < 2")
    if not sparse.isspmatrix_csr(A) or A.shape != (b.size, b.size):
        raise ValueError("A must be a square CSR matrix matching b")
    diagonal = A.diagonal()
    if np.any(diagonal == 0):
        raise ValueError("A has a zero diagonal element")
    if u_star is not None and u_star.shape != b.shape:
        raise ValueError("u_star must have the same shape as b")

    x = np.zeros_like(b, dtype=float)
    norm_b = np.linalg.norm(b)
    norm_u = None if u_star is None else np.linalg.norm(u_star)
    if norm_u is not None and norm_u == 0:
        raise ValueError("u_star must have nonzero norm")
    if norm_b == 0:
        return Result(x, 0, [0.0], None if u_star is None else [1.0], 0.0, True)
    residuals = [1.0]
    deltas = None if u_star is None else [float(np.linalg.norm(u_star) / norm_u)]
    if method != "jacobi":
        # A = D+L+U. This sparse triangular solve is a sequential sweep:
        # (D+omega*L)x_new = omega*b + ((1-omega)*D-omega*U)x_old.
        relaxation = 1.0 if method == "gauss_seidel" else omega
        lower = (sparse.diags(diagonal) + relaxation * sparse.tril(A, -1)).tocsr()
        upper = sparse.triu(A, 1, format="csr")
        scaled_b = relaxation * b
    start = perf_counter()
    for iteration in range(1, max_iter + 1):
        if method == "jacobi":
            x = x + (b - A @ x) / diagonal
        else:
            rhs = scaled_b + (1.0 - relaxation) * diagonal * x - relaxation * (upper @ x)
            x = spsolve_triangular(lower, rhs, lower=True)
        eta = float(np.linalg.norm(b - A @ x) / norm_b)
        residuals.append(eta)
        if deltas is not None:
            deltas.append(float(np.linalg.norm(u_star - x) / norm_u))
        if eta <= tol:
            return Result(x, iteration, residuals, deltas, perf_counter() - start, True)
    return Result(x, max_iter, residuals, deltas, perf_counter() - start, False)


def search_omega(A: sparse.csr_matrix, b: np.ndarray, tol: float, max_iter: int):
    """Scan 11 values on [1,1.95], then use step 0.01 around the best."""
    cache: dict[float, Result] = {}

    def evaluate(value: float) -> Result:
        value = round(value, 2)
        if value not in cache:
            cache[value] = solve_stationary(A, b, "sor", tol=tol, max_iter=max_iter, omega=value)
        return cache[value]

    coarse = [round(1.0 + 0.1 * i, 2) for i in range(10)] + [1.95]
    for value in coarse:
        evaluate(value)
    successful = [value for value in coarse if cache[value].converged]
    if not successful:
        raise RuntimeError("No coarse SOR trial met the residual target; increase max_iter")
    coarse_best = min(successful, key=lambda value: (cache[value].iterations, value))
    refined = sorted({round(coarse_best + 0.01 * i, 2) for i in range(-5, 6)
                      if 1.0 <= round(coarse_best + 0.01 * i, 2) <= 1.95})
    for value in refined:
        evaluate(value)
    successful = [value for value, result in cache.items() if result.converged]
    best = min(successful, key=lambda value: (cache[value].iterations, value))
    return coarse, refined, coarse_best, best, cache


def run_experiment(tol: float, max_iter: int, plot: Path | None,
                   solutions: Path | None, summary_csv: Path | None) -> str:
    labels = {"jacobi": "Якоби", "gauss_seidel": "Зейдель", "sor": "ПВР (SOR)"}
    lines = [
        "Лабораторная работа №1, этап 8.2, вариант №11 = R1",
        "N — число узлов по каждой оси, включая границу; неизвестных (N-2)^2.",
        f"x0=0; критерий (14): η_s=||b-Ax^(s)||2/||b||2 <= ε_res={tol:.1e}; максимум итераций: {max_iter}",
        "n_s — число итераций до критерия; истинная невязка пересчитывается на каждой итерации.",
        "δ_s=||u*-x^(s)||2/||u*||2 вычисляется только для теста с известным u*.",
        "Время зависит от компьютера и реализации.",
    ]
    rows: list[dict[str, object]] = []

    # Section 8.2 test: prescribed exact vector on N=32, b*=A u*.
    A32, _ = build_system(32, 32)
    n = A32.shape[0]
    m = np.arange(n)
    u_star = 1.0 + np.sin(2.0 * np.pi * m / n) + 0.1 * np.cos(6.0 * np.pi * m / n)
    b_star = A32 @ u_star
    test_results = {method: solve_stationary(A32, b_star, method, tol=tol,
                                             max_iter=max_iter, omega=1.5, u_star=u_star)
                    for method in labels}
    lines.extend(["", "Проверочная система N=32: u*_m=1+sin(2πm/n)+0.1cos(6πm/n), b*=Au*; ПВР ω=1.5",
                  "Метод              n_s           η_s           δ_s  Время, с  Статус"])
    for method, result in test_results.items():
        lines.append(f"{labels[method]:<14} {result.iterations:>8}  {result.eta:>12.3e}  "
                     f"{result.delta:>12.3e}  {result.seconds:>8.3f}  "
                     f"{'сошёлся' if result.converged else 'не сошёлся'}")
        rows.append(dict(case="test", N=32, method=method, omega=1.5 if method == "sor" else "",
                         n_s=result.iterations, eta_s=result.eta, delta_s=result.delta,
                         epsilon_res=tol, seconds=result.seconds, converged=result.converged))
    gs = test_results["gauss_seidel"]
    sor_one = solve_stationary(A32, b_star, "sor", tol=tol, max_iter=max_iter,
                               omega=1.0, u_star=u_star)
    if gs.iterations != sor_one.iterations or not np.allclose(gs.x, sor_one.x, rtol=0, atol=1e-12):
        raise RuntimeError("SOR with omega=1 does not agree with Gauss-Seidel")
    lines.append("Дополнительная проверка: ПВР при ω=1 совпадает с методом Зейделя.")

    # Assigned R1 right-hand side on the required N=32 and N=64 grids.
    A64, b64 = build_system(64, 64)
    coarse, refined, coarse_best, best, search = search_omega(A64, b64, tol, max_iter)
    lines.extend(["", "Поиск ПВР для варианта R1, N=64: 11 значений ω в [1,1.95], затем шаг 0.01",
                  "Грубый поиск:"])
    for phase, values in (("coarse", coarse), ("refined", refined)):
        if phase == "refined":
            lines.extend([f"Лучшее значение грубого поиска: ω={coarse_best:.2f}",
                          "Уточнение в его окрестности:"])
        for value in values:
            result = search[value]
            lines.append(f"  ω={value:.2f}: n_s={result.iterations:5d}; η_s={result.eta:.3e}; "
                         f"время={result.seconds:.3f} с; "
                         f"{'сошёлся' if result.converged else 'не сошёлся'}")
            rows.append(dict(case=f"omega_{phase}", N=64, method="sor", omega=value,
                             n_s=result.iterations, eta_s=result.eta, delta_s="",
                             epsilon_res=tol, seconds=result.seconds, converged=result.converged))
    lines.append(f"Наилучшее среди проверенных значений: ω={best:.2f}, n_s={search[best].iterations}.")

    plot_results: dict[str, Result] = {}
    solution_arrays: dict[str, np.ndarray] = {}
    for grid in (32, 64):
        A, b = (A64, b64) if grid == 64 else build_system(32, 32)
        lines.extend(["", f"Вариант R1, N={grid}: n={b.size}, nnz={A.nnz}; ПВР ω={best:.2f}",
                      "Метод              n_s           η_s  Время, с  Статус"])
        for method in labels:
            result = (search[best] if grid == 64 and method == "sor" else
                      solve_stationary(A, b, method, tol=tol, max_iter=max_iter, omega=best))
            lines.append(f"{labels[method]:<14} {result.iterations:>8}  "
                         f"{result.eta:>12.3e}  {result.seconds:>8.3f}  "
                         f"{'сошёлся' if result.converged else 'не сошёлся'}")
            rows.append(dict(case="R1", N=grid, method=method, omega=best if method == "sor" else "",
                             n_s=result.iterations, eta_s=result.eta, delta_s="",
                             epsilon_res=tol, seconds=result.seconds, converged=result.converged))
            solution_arrays[f"N{grid}_{method}"] = result.x.reshape(grid - 2, grid - 2)
            if grid == 64:
                plot_results[method] = result
    if solutions is not None:
        np.savez_compressed(solutions, **solution_arrays)
        lines.extend(["", f"Массивы решений варианта R1: {solutions}"])
    if summary_csv is not None:
        with summary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case", "N", "method", "omega", "n_s",
                                                       "eta_s", "delta_s", "epsilon_res", "seconds", "converged"])
            writer.writeheader()
            writer.writerows(rows)
        lines.append(f"Сводная таблица CSV: {summary_csv}")
    if plot is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for method, result in plot_results.items():
            axes[0].semilogy(range(len(result.residuals)), result.residuals, label=labels[method])
        axes[0].axhline(tol, color="black", linestyle="--", linewidth=1, label="Порог")
        axes[0].set(xlabel="Итерация", ylabel="η_s", title="Истинная невязка, R1, N=64")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
        values = sorted(search)
        axes[1].plot(values, [search[value].iterations for value in values], "o-")
        axes[1].axvline(best, color="red", linestyle="--", label=f"Лучшее проверенное ω={best:.2f}")
        axes[1].set(xlabel="ω", ylabel="Итераций до критерия", title="Поиск параметра ПВР, R1, N=64")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(plot, dpi=160)
        plt.close(fig)
        lines.extend(["", f"График: {plot}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=100_000)
    parser.add_argument("--plot", type=Path, default=Path(__file__).with_name("lab1_stage_8_2_R1_convergence.png"))
    parser.add_argument("--solutions", type=Path, default=Path(__file__).with_name("lab1_stage_8_2_R1_solutions.npz"))
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("lab1_stage_8_2_R1_summary.csv"))
    args = parser.parse_args()
    print(run_experiment(args.tol, args.max_iter, args.plot, args.solutions, args.csv))


if __name__ == "__main__":
    main()
