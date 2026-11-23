"""Lab 1, section 8.3: CG and PCG for variant R1.

CG/PCG recurrences are implemented here. All methods start at zero and stop
when the directly recomputed residual ||b-Ax||_2/||b||_2 <= epsilon_res.
The SSOR preconditioner is applied by two sparse triangular solves; P^-1 is
never formed. A trial run precedes three timed runs for each reported case.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve_triangular

from lab1_stage_8_1_R1 import build_system


@dataclass
class CGResult:
    x: np.ndarray
    iterations: int
    matvecs: int
    residuals: list[float]
    iteration_seconds: float
    converged: bool

    @property
    def eta(self) -> float:
        return self.residuals[-1]


def csr_memory_bytes(A: sparse.csr_matrix) -> int:
    return A.data.nbytes + A.indices.nbytes + A.indptr.nbytes


def make_preconditioner(
    A: sparse.csr_matrix, kind: str, omega: float = 1.0
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Return z=P^-1 r as an action, plus preparation time in seconds."""
    start = perf_counter()
    if kind == "none":
        return lambda r: r.copy(), 0.0
    diagonal = A.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("preconditioning requires a positive diagonal")
    if kind == "diagonal":
        return lambda r: r / diagonal, perf_counter() - start
    if kind != "ssor":
        raise ValueError("kind must be none, diagonal or ssor")
    if not 0 < omega < 2:
        raise ValueError("SSOR requires 0 < omega < 2")
    asymmetry = sparse.linalg.norm(A - A.T) / sparse.linalg.norm(A)
    if asymmetry > 1e-12:
        raise ValueError("SSOR requires a symmetric matrix")
    lower = sparse.tril(A, -1, format="csr")
    B = (sparse.diags(diagonal) + omega * lower).tocsr()
    BT = B.T.tocsr()
    scale = omega * (2.0 - omega)

    def apply(r: np.ndarray) -> np.ndarray:
        y = spsolve_triangular(B, r, lower=True)
        z0 = spsolve_triangular(BT, diagonal * y, lower=False)
        return scale * z0

    return apply, perf_counter() - start


def solve_cg(
    A: sparse.csr_matrix,
    b: np.ndarray,
    apply_preconditioner: Callable[[np.ndarray], np.ndarray],
    *,
    tol: float = 1e-8,
    max_iter: int = 100_000,
) -> CGResult:
    """Own CG/PCG recurrence; count Ap and each explicit A*x residual check."""
    if not sparse.isspmatrix_csr(A) or A.shape != (b.size, b.size):
        raise ValueError("A must be a square CSR matrix matching b")
    if not 0 < tol < 1 or max_iter < 1:
        raise ValueError("tol must be in (0,1), max_iter positive")
    x = np.zeros_like(b, dtype=float)
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return CGResult(x, 0, 0, [0.0], 0.0, True)
    r = b.astype(float, copy=True)  # x0=0, hence r0=b without a matvec.
    z = apply_preconditioner(r)
    rho = float(r @ z)
    if not np.isfinite(rho) or rho <= 0:
        raise ArithmeticError("r^T P^-1 r must be positive")
    p = z.copy()
    residuals = [1.0]
    matvecs = 0
    start = perf_counter()
    for iteration in range(1, max_iter + 1):
        q = A @ p
        matvecs += 1
        denominator = float(p @ q)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ArithmeticError("p^T A p must be positive")
        alpha = rho / denominator
        x += alpha * p
        r -= alpha * q
        # The stopping rule always uses the true, directly recomputed residual.
        eta = float(np.linalg.norm(b - A @ x) / norm_b)
        matvecs += 1
        residuals.append(eta)
        if eta <= tol:
            return CGResult(x, iteration, matvecs, residuals, perf_counter() - start, True)
        z = apply_preconditioner(r)
        rho_new = float(r @ z)
        if not np.isfinite(rho_new) or rho_new <= 0:
            raise ArithmeticError("new r^T P^-1 r must be positive")
        p = z + (rho_new / rho) * p
        rho = rho_new
    return CGResult(x, max_iter, matvecs, residuals, perf_counter() - start, False)


def run_once(A: sparse.csr_matrix, b: np.ndarray, kind: str, omega: float,
             tol: float, max_iter: int) -> tuple[CGResult, float]:
    apply, prep_seconds = make_preconditioner(A, kind, omega)
    return solve_cg(A, b, apply, tol=tol, max_iter=max_iter), prep_seconds


def benchmark(A: sparse.csr_matrix, b: np.ndarray, kind: str, omega: float,
              tol: float, max_iter: int) -> tuple[CGResult, float, float]:
    """One untimed trial, then medians of 3 preparation and iteration timings."""
    trial, _ = run_once(A, b, kind, omega, tol, max_iter)
    preparations: list[float] = []
    iterations: list[float] = []
    result = trial
    for _ in range(3):
        result, prep = run_once(A, b, kind, omega, tol, max_iter)
        if result.iterations != trial.iterations or result.matvecs != trial.matvecs:
            raise RuntimeError("repeated runs produced inconsistent iteration counts")
        preparations.append(prep)
        iterations.append(result.iteration_seconds)
    return result, median(preparations), median(iterations)


def run_experiment(tol: float, max_iter: int, plot: Path | None,
                   solutions: Path | None, summary_csv: Path | None) -> str:
    names = {"none": "CG", "diagonal": "PCG-D", "ssor": "PCG-SSOR"}
    lines = [
        "Лабораторная работа №1, этап 8.3, вариант №11 = R1",
        f"x0=0; ε_res={tol:.1e}; критерий η_s=||b-Ax^(s)||2/||b||2 <= ε_res; максимум итераций: {max_iter}",
        "Число matvec включает Ap и прямой пересчёт A*x для истинной невязки на каждой итерации.",
        "Время подготовки и итераций: медиана 3 измерений после 1 пробного запуска.",
        "PCG-D: P=D; PCG-SSOR: P=B D^-1 B^T/[ω(2-ω)], B=D+ωL; P^-1 не формируется.",
    ]
    records: list[dict[str, object]] = []
    solution_arrays: dict[str, np.ndarray] = {}
    plot_results: dict[str, CGResult] = {}
    iteration_counts: dict[str, list[int]] = {kind: [] for kind in names}

    # The prescribed manufactured test from section 8.2.
    A32, b32 = build_system(32, 32)
    n = A32.shape[0]
    m = np.arange(n)
    u_star = 1.0 + np.sin(2.0 * np.pi * m / n) + 0.1 * np.cos(6.0 * np.pi * m / n)
    b_star = A32 @ u_star
    lambda_min = float(eigsh(A32, k=1, which="SA", return_eigenvectors=False)[0])
    lambda_max = float(eigsh(A32, k=1, which="LA", return_eigenvectors=False)[0])
    kappa_est = lambda_max / lambda_min
    lines.extend(["", "Проверочная система N=32: u*_m=1+sin(2πm/n)+0.1cos(6πm/n), b*=Au*",
                  "В тесте PCG-SSOR использует ω=1.00; параметр основного эксперимента подбирается отдельно.",
                  f"Оценка обусловленности μ₂(A)≈{kappa_est:.3e} (крайние собственные значения).",
                  "Метод          n_s  matvec           η_s           δ_s        μ₂η_s  prep, с  iter, с"])
    for kind in names:
        # A conservative, separate SSOR choice for the test only; main choice follows below.
        test_omega = 1.0
        result, prep, elapsed = benchmark(A32, b_star, kind, test_omega, tol, max_iter)
        delta = float(np.linalg.norm(u_star - result.x) / np.linalg.norm(u_star))
        bound = kappa_est * result.eta
        if not result.converged or delta > bound * (1.0 + 1e-6):
            raise RuntimeError("test residual/error check failed")
        lines.append(f"{names[kind]:<10} {result.iterations:>6}  {result.matvecs:>6}  "
                     f"{result.eta:>12.3e}  {delta:>12.3e}  {bound:>12.3e}  "
                     f"{prep:>7.3f}  {elapsed:>7.3f}")
        records.append(dict(case="test", N=32, n=n, nnz=A32.nnz, method=kind,
                            omega=test_omega if kind == "ssor" else "", iterations=result.iterations,
                            matvecs=result.matvecs, prep_seconds=prep, iter_seconds=elapsed,
                            eta=result.eta, delta=delta, kappa_eta_bound=bound,
                            csr_bytes=csr_memory_bytes(A32), epsilon_res=tol, converged=result.converged))
    lines.append("Тест: η_s <= ε_res и δ_s <= μ₂(A)η_s для всех трёх методов.")

    # SSOR parameter is selected independently of the SOR iteration parameter.
    A64, b64 = build_system(64, 64)
    ssor_candidates = [1.0, 1.2, 1.4, 1.6, 1.8, 1.9]
    ssor_trials: dict[float, CGResult] = {}
    lines.extend(["", "Отдельный подбор ω для предобуславливателя SSOR на R1, N=64:"])
    for omega in ssor_candidates:
        result, _ = run_once(A64, b64, "ssor", omega, tol, max_iter)
        ssor_trials[omega] = result
        lines.append(f"  ω={omega:.2f}: {result.iterations} ит.; η_s={result.eta:.3e}; "
                     f"{'сошёлся' if result.converged else 'не сошёлся'}")
    successful = [omega for omega, result in ssor_trials.items() if result.converged]
    if not successful:
        raise RuntimeError("no SSOR parameter met the residual target")
    best_omega = min(successful, key=lambda omega: (ssor_trials[omega].iterations, omega))
    lines.append(f"Выбрано ω_SSOR={best_omega:.2f}: наименьшее число итераций среди отдельно проверенных значений.")
    lines.append("ω_SSOR относится к предобуславливателю; параметр ПВР из этапа 8.2 здесь не использован.")

    for N in (32, 64, 128):
        A, b = (A32, b32) if N == 32 else (
            (A64, b64) if N == 64 else build_system(128, 128))
        memory = csr_memory_bytes(A)
        lines.extend(["", f"Вариант R1, N={N}: n={b.size}, nnz={A.nnz}, CSR={memory} байт",
                      "Метод          n_s  matvec           η_s  prep, с  iter, с  Статус"])
        for kind in names:
            result, prep, elapsed = benchmark(A, b, kind, best_omega, tol, max_iter)
            lines.append(f"{names[kind]:<10} {result.iterations:>6}  {result.matvecs:>6}  "
                         f"{result.eta:>12.3e}  {prep:>7.3f}  {elapsed:>7.3f}  "
                         f"{'сошёлся' if result.converged else 'не сошёлся'}")
            records.append(dict(case="R1", N=N, n=b.size, nnz=A.nnz, method=kind,
                                omega=best_omega if kind == "ssor" else "", iterations=result.iterations,
                                matvecs=result.matvecs, prep_seconds=prep, iter_seconds=elapsed,
                                eta=result.eta, delta="", kappa_eta_bound="", csr_bytes=memory,
                                epsilon_res=tol, converged=result.converged))
            solution_arrays[f"N{N}_{kind}"] = result.x.reshape(N - 2, N - 2)
            iteration_counts[kind].append(result.iterations)
            if N == 128:
                plot_results[kind] = result
    if solutions is not None:
        np.savez_compressed(solutions, **solution_arrays)
        lines.extend(["", f"Массивы решений: {solutions}"])
    if summary_csv is not None:
        with summary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case", "N", "n", "nnz", "method", "omega",
                                                       "iterations", "matvecs", "prep_seconds", "iter_seconds",
                                                       "eta", "delta", "kappa_eta_bound", "csr_bytes",
                                                       "epsilon_res", "converged"])
            writer.writeheader()
            writer.writerows(records)
        lines.append(f"Сводная таблица CSV: {summary_csv}")
    if plot is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for kind, result in plot_results.items():
            axes[0].semilogy(range(len(result.residuals)), result.residuals, label=names[kind])
        axes[0].axhline(tol, color="black", linestyle="--", linewidth=1, label="Порог")
        axes[0].set(xlabel="Итерация", ylabel=r"$\eta_s$", title="Истинная невязка, R1, N=128")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
        for kind, counts in iteration_counts.items():
            axes[1].plot([32, 64, 128], counts, "o-", label=names[kind])
        axes[1].set(xlabel="N, узлов по оси", ylabel="Итераций", title="Число итераций от N")
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
    parser.add_argument("--plot", type=Path, default=Path(__file__).with_name("lab1_stage_8_3_R1_convergence.png"))
    parser.add_argument("--solutions", type=Path, default=Path(__file__).with_name("lab1_stage_8_3_R1_solutions.npz"))
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("lab1_stage_8_3_R1_summary.csv"))
    args = parser.parse_args()
    print(run_experiment(args.tol, args.max_iter, args.plot, args.solutions, args.csv))


if __name__ == "__main__":
    main()
