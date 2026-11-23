"""Lab 1, section 8.4: independent verification of R1 linear systems.

Compare the saved own CG/PCG solutions with sparse direct spsolve and SciPy CG.
Check every control solution by a newly computed true normalized residual.
Agreement verifies the same discrete Ax=b, not the PDE discretization error.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, cg, spsolve

from lab1_stage_8_1_R1 import build_system


METHODS = ("none", "diagonal", "ssor")
LABELS = {"none": "CG", "diagonal": "PCG-D", "ssor": "PCG-SSOR"}


def true_residual(A, b: np.ndarray, x: np.ndarray) -> float:
    """Return ||b-Ax||_2 / ||b||_2 without using a solver's internal residual."""
    norm_b = np.linalg.norm(b)
    return float(np.linalg.norm(b - A @ x) / norm_b) if norm_b else float(np.linalg.norm(A @ x))


def relative_errors(x: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Relative Euclidean and infinity norms of the difference."""
    norm2 = np.linalg.norm(reference)
    norm_inf = np.linalg.norm(reference, ord=np.inf)
    if norm2 == 0 or norm_inf == 0:
        raise ValueError("reference solution has zero norm")
    difference = x - reference
    return (float(np.linalg.norm(difference) / norm2),
            float(np.linalg.norm(difference, ord=np.inf) / norm_inf))


def verify(grids: list[int], saved_solutions: Path, library_rtol: float,
           max_iter: int, reference_output: Path | None) -> str:
    if not 0 < library_rtol < 1e-8:
        raise ValueError("library CG rtol must be stricter than the own solvers' 1e-8")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not saved_solutions.is_file():
        raise FileNotFoundError(saved_solutions)

    lines = [
        "Лабораторная работа №1, этап 8.4, вариант №11 = R1",
        "Контроль: scipy.sparse.linalg.spsolve и scipy.sparse.linalg.cg.",
        f"Библиотечный CG: rtol={library_rtol:.1e}, atol=0, maxiter={max_iter}.",
        "Все η=||b-Ax||₂/||b||₂ пересчитаны непосредственно после завершения решателя.",
        "L₂ и L∞ — относительные нормы разности с указанным контрольным решением.",
    ]
    references: dict[str, np.ndarray] = {}
    with np.load(saved_solutions) as own:
        for N in grids:
            if N < 3:
                raise ValueError("N must be at least 3")
            A, b = build_system(N, N)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", MatrixRankWarning)
                direct = spsolve(A, b)
            rank_warnings = [w for w in caught if issubclass(w.category, MatrixRankWarning)]
            if rank_warnings or not np.all(np.isfinite(direct)):
                raise RuntimeError(f"spsolve failed on N={N}: rank warning or nonfinite result")
            eta_direct = true_residual(A, b, direct)
            if eta_direct > library_rtol:
                raise RuntimeError(f"spsolve true residual too large on N={N}: {eta_direct}")

            library_iterations = 0
            def count_iteration(_x: np.ndarray) -> None:
                nonlocal library_iterations
                library_iterations += 1

            library, info = cg(A, b, x0=np.zeros_like(b), rtol=library_rtol,
                               atol=0.0, maxiter=max_iter, callback=count_iteration)
            eta_library = true_residual(A, b, library)
            if info != 0 or not np.all(np.isfinite(library)) or eta_library > library_rtol:
                raise RuntimeError(f"library CG failed on N={N}: info={info}, eta={eta_library}")
            direct_library_l2, direct_library_linf = relative_errors(library, direct)
            references[f"N{N}_direct"] = direct.reshape(N - 2, N - 2)
            references[f"N{N}_scipy_cg"] = library.reshape(N - 2, N - 2)

            lines.extend([
                "",
                f"Сетка N={N}: n={A.shape[0]}, nnz={A.nnz}",
                f"Прямое решение: статус=успешно, η={eta_direct:.3e}",
                f"Библиотечный CG: info={info}, итераций={library_iterations}, η={eta_library:.3e}",
                f"Библиотечный CG против прямого: L₂={direct_library_l2:.3e}, L∞={direct_library_linf:.3e}",
                "Собственный метод    η                 L₂/прямое      L∞/прямое      L₂/CG           L∞/CG",
            ])
            for method in METHODS:
                key = f"N{N}_{method}"
                if key not in own:
                    raise KeyError(f"saved solution missing: {key}")
                x = own[key].ravel()
                if x.shape != b.shape or not np.all(np.isfinite(x)):
                    raise ValueError(f"invalid saved solution: {key}")
                eta_own = true_residual(A, b, x)
                if eta_own > 1e-8:
                    raise RuntimeError(f"own solution does not meet stage 8.3 tolerance: {key}")
                l2_direct, linf_direct = relative_errors(x, direct)
                l2_library, linf_library = relative_errors(x, library)
                lines.append(f"{LABELS[method]:<19} {eta_own:.3e}       {l2_direct:.3e}       "
                             f"{linf_direct:.3e}       {l2_library:.3e}       {linf_library:.3e}")
    if reference_output is not None:
        np.savez_compressed(reference_output, **references)
        lines.extend(["", f"Контрольные массивы: {reference_output}"])
    lines.extend(["", "Согласие решателей подтверждает решение одной и той же СЛАУ Ax=b.",
                  "Погрешность дискретизации исходной краевой задачи здесь не оценивается."])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--solutions", type=Path,
                        default=Path(__file__).with_name("lab1_stage_8_3_R1_solutions.npz"))
    parser.add_argument("--library-rtol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=100_000)
    parser.add_argument("--reference-output", type=Path,
                        default=Path(__file__).with_name("lab1_stage_8_4_R1_controls.npz"))
    args = parser.parse_args()
    print(verify(args.grids, args.solutions, args.library_rtol,
                 args.max_iter, args.reference_output))


if __name__ == "__main__":
    main()
