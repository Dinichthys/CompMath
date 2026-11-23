"""Lab 1, section 8.5: summary table, required figures, and R1 report.

Reads the saved results from stages 8.1-8.4, recomputes convergence traces
and condition estimates, and writes a self-contained Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

from lab1_stage_8_1_R1 import build_system
from lab1_stage_8_2_R1 import solve_stationary
from lab1_stage_8_3_R1 import make_preconditioner, solve_cg, csr_memory_bytes
from lab1_stage_8_4_R1 import true_residual, relative_errors


FIELDNAMES = ["stage", "case", "variant", "N", "n", "nnz", "method", "omega",
              "iterations", "matvecs", "prep_seconds", "solve_seconds", "eta",
              "delta", "epsilon_res", "csr_bytes", "status"]
METHOD_NAMES = {"jacobi": "Якоби", "gauss_seidel": "Зейдель", "sor": "ПВР",
                "none": "CG", "diagonal": "PCG-D", "ssor": "PCG-SSOR"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(row) + " |" for row in rows)])


def run(project_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = (project_dir / "lab1_stage_8_1_R1_results.txt").read_text(encoding="utf-8")
    stage2 = read_csv(project_dir / "lab1_stage_8_2_R1_summary.csv")
    stage3 = read_csv(project_dir / "lab1_stage_8_3_R1_summary.csv")
    stage3_report = (project_dir / "lab1_stage_8_3_R1_results.txt").read_text(encoding="utf-8")
    stage4_report = (project_dir / "lab1_stage_8_4_R1_results.txt").read_text(encoding="utf-8")
    if "Уравнение строки m=14:" not in stage1:
        raise RuntimeError("stage 8.1 row equation is missing")
    omega_sor = min((r for r in stage2 if r["case"].startswith("omega_") and r["converged"] == "True"),
                    key=lambda r: (int(r["n_s"]), float(r["omega"])))
    sor_value = float(omega_sor["omega"])
    ssor_match = re.search(r"Выбрано ω_SSOR=([0-9.]+)", stage3_report)
    if ssor_match is None:
        raise RuntimeError("SSOR parameter is missing from stage 8.3 results")
    ssor_value = float(ssor_match.group(1))
    A_by_N, b_by_N, kappa_by_N = {}, {}, {}
    for N in (32, 64, 128):
        A, b = build_system(N, N)
        A_by_N[N], b_by_N[N] = A, b
        lambda_min = float(eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0])
        lambda_max = float(eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-8)[0])
        kappa_by_N[N] = lambda_max / lambda_min

    # Consolidate the actual test and R1 runs. Tuning trials remain in their
    # original stage 8.2 protocol and stage 8.3 protocol.
    records: list[dict[str, object]] = []
    for row in stage2:
        if row["case"] not in {"test", "R1"}:
            continue
        N = int(row["N"])
        A = A_by_N[N]
        records.append(dict(stage="8.2", case=row["case"], variant="R1", N=N,
                            n=A.shape[0], nnz=A.nnz, method=row["method"], omega=row["omega"],
                            iterations=row["n_s"], matvecs="", prep_seconds="",
                            solve_seconds=row["seconds"], eta=row["eta_s"], delta=row["delta_s"],
                            epsilon_res=row["epsilon_res"], csr_bytes=csr_memory_bytes(A),
                            status=row["converged"]))
    for row in stage3:
        N = int(row["N"])
        A = A_by_N[N]
        records.append(dict(stage="8.3", case=row["case"], variant="R1", N=N,
                            n=A.shape[0], nnz=A.nnz, method=row["method"], omega=row["omega"],
                            iterations=row["iterations"], matvecs=row["matvecs"],
                            prep_seconds=row["prep_seconds"], solve_seconds=row["iter_seconds"],
                            eta=row["eta"], delta=row["delta"], epsilon_res=row["epsilon_res"],
                            csr_bytes=row["csr_bytes"], status=row["converged"]))

    # Verify and summarize both independent controls from section 8.4.
    info_rows = re.findall(r"Библиотечный CG: info=(\d+), итераций=(\d+), η=([0-9.eE+-]+)", stage4_report)
    if len(info_rows) != 3 or any(info != "0" for info, _, _ in info_rows):
        raise RuntimeError("stage 8.4 library CG return codes are incomplete")
    control_rows = []
    error_rows = []
    with np.load(project_dir / "lab1_stage_8_3_R1_solutions.npz") as own, \
         np.load(project_dir / "lab1_stage_8_4_R1_controls.npz") as controls:
        for index, N in enumerate((32, 64, 128)):
            A, b = A_by_N[N], b_by_N[N]
            direct = controls[f"N{N}_direct"].ravel()
            library = controls[f"N{N}_scipy_cg"].ravel()
            direct_eta = true_residual(A, b, direct)
            library_eta = true_residual(A, b, library)
            if direct_eta > 1e-10 or library_eta > 1e-10:
                raise RuntimeError(f"control residual failed on N={N}")
            control_rows.append([str(N), f"{direct_eta:.3e}", str(info_rows[index][1]), f"{library_eta:.3e}"])
            for method, vector in (("direct", direct), ("scipy_cg", library)):
                records.append(dict(stage="8.4", case="R1", variant="R1", N=N,
                                    n=A.shape[0], nnz=A.nnz, method=method, omega="",
                                    iterations=info_rows[index][1] if method == "scipy_cg" else "",
                                    matvecs="", prep_seconds="", solve_seconds="",
                                    eta=library_eta if method == "scipy_cg" else direct_eta,
                                    delta="", epsilon_res=1e-10, csr_bytes=csr_memory_bytes(A),
                                    status=True))
            for method in ("none", "diagonal", "ssor"):
                x = own[f"N{N}_{method}"].ravel()
                l2, linf = relative_errors(x, direct)
                error_rows.append([str(N), METHOD_NAMES[method], f"{l2:.3e}", f"{linf:.3e}"])
                if true_residual(A, b, x) > 1e-8:
                    raise RuntimeError(f"own solution failed on N={N}, method={method}")
            if N == 128:
                field = own["N128_ssor"].copy()

    summary_path = output_dir / "lab1_stage_8_5_R1_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    # Recompute histories to make the required logarithmic residual plots.
    A64, b64 = A_by_N[64], b_by_N[64]
    stationary = {method: solve_stationary(A64, b64, method, omega=sor_value)
                  for method in ("jacobi", "gauss_seidel", "sor")}
    A128, b128 = A_by_N[128], b_by_N[128]
    krylov = {}
    for method in ("none", "diagonal", "ssor"):
        apply, _ = make_preconditioner(A128, method, ssor_value)
        krylov[method] = solve_cg(A128, b128, apply)
    for result in [*stationary.values(), *krylov.values()]:
        if not result.converged or result.residuals[-1] > 1e-8:
            raise RuntimeError("a convergence trace did not reach epsilon_res")

    residual_figure = output_dir / "lab1_stage_8_5_R1_residuals.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title in ((axes[0], stationary, "Стационарные методы, N=64"),
                            (axes[1], krylov, "CG и PCG, N=128")):
        for method, result in data.items():
            ax.semilogy(range(len(result.residuals)), result.residuals, label=METHOD_NAMES[method])
        ax.axhline(1e-8, color="black", linestyle="--", linewidth=1, label="Порог 10⁻⁸")
        ax.set(xlabel="Итерация s", ylabel=r"$\eta_s=\|b-Ax^{(s)}\|_2/\|b\|_2$", title=title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(residual_figure, dpi=160)
    plt.close(fig)

    iterations_figure = output_dir / "lab1_stage_8_5_R1_iterations.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, stage, methods, title in ((axes[0], "8.2", ("jacobi", "gauss_seidel", "sor"), "Стационарные методы"),
                                      (axes[1], "8.3", ("none", "diagonal", "ssor"), "CG и PCG")):
        for method in methods:
            subset = sorted((r for r in records if r["stage"] == stage and r["case"] == "R1"
                             and r["method"] == method), key=lambda r: int(r["N"]))
            ax.plot([int(r["N"]) for r in subset], [int(r["iterations"]) for r in subset],
                    "o-", label=METHOD_NAMES[method])
        ax.set(xlabel="N, узлов по оси", ylabel="Число итераций", title=title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(iterations_figure, dpi=160)
    plt.close(fig)

    field_figure = output_dir / "lab1_stage_8_5_R1_field_N128.png"
    full = np.zeros((128, 128))
    full[1:-1, 1:-1] = field
    peak_index = np.unravel_index(np.argmax(full), full.shape)
    peak_x, peak_y = peak_index[1] / 127, peak_index[0] / 127
    peak_value = float(full[peak_index])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(full, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", interpolation="nearest")
    ax.plot(peak_x, peak_y, "r+", markersize=11, markeredgewidth=1.8, label="Максимум")
    ax.set(xlabel="x", ylabel="y", title="Концентрация вещества u(x,y), R1, N=128")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Концентрация u")
    fig.tight_layout()
    fig.savefig(field_figure, dpi=160)
    plt.close(fig)

    mass_by_N = {}
    with np.load(project_dir / "lab1_stage_8_3_R1_solutions.npz") as own:
        for N in (32, 64, 128):
            mass_by_N[N] = float(own[f"N{N}_ssor"].sum() / (N - 1) ** 2)

    stage2_test = [r for r in records if r["stage"] == "8.2" and r["case"] == "test"]
    stage3_test = [r for r in records if r["stage"] == "8.3" and r["case"] == "test"]
    stage2_main = [r for r in records if r["stage"] == "8.2" and r["case"] == "R1"]
    stage3_main = [r for r in records if r["stage"] == "8.3" and r["case"] == "R1"]
    format_main = lambda rows: md_table(
        ["N", "Метод", "Итераций", "Порог η", "Достигнутая ηₛ", "Время итераций, с"],
        [[str(r["N"]), METHOD_NAMES[r["method"]], str(r["iterations"]), f"{float(r['epsilon_res']):.0e}",
          f"{float(r['eta']):.3e}", f"{float(r['solve_seconds']):.3f}"] for r in rows])
    condition_table = md_table(
        ["N", "n", "nnz(A)", "Оценка μ₂(A)", "Масса вещества"],
        [[str(N), str(A_by_N[N].shape[0]), str(A_by_N[N].nnz), f"{kappa_by_N[N]:.1f}",
          f"{mass_by_N[N]:.9f}"] for N in (32, 64, 128)])
    max_l2 = max(float(row[2]) for row in error_rows)
    max_linf = max(float(row[3]) for row in error_rows)
    report = f"""# Лабораторная работа №1 — вариант №11 (R1)

**Итерационные методы решения разреженных СЛАУ.** Исследуется диффузия вещества с поглощением в единичном квадрате:

$$
-\\nabla\\cdot(a\\nabla u)+cu=f,\\qquad u|_{{\\partial\\Omega}}=0.
$$

Для R1: $a=1+0.30\\sin(2\\pi x)\\sin(2\\pi y)$, $c=1$, $f=10\\exp(-((x-0.5)^2+(y-0.5)^2)/0.02)$. Сетка содержит граничные узлы, число внутренних неизвестных $n=(N-2)^2$. Матрица хранится в CSR. Все итерационные методы стартуют с нуля и останавливаются при истинной нормированной невязке $\\eta_s=\\|b-Ax^{{(s)}}\\|_2/\\|b\\|_2\\leq10^{{-8}}$; превышение максимума итераций отмечается как несходимость.

## 8.1. Сборка матрицы

При N=8 получена матрица 36×36 с 156 ненулевыми элементами, не более 5 в строке. Относительная мера нарушения симметрии равна 2.654·10⁻¹⁷; минимальный диагональный элемент 151.632808. Для постоянного, знакопеременного и синусоидального векторов значения $z^TAz$ положительны (1212.000, 12972.000 и 250.024). Разложение Холецкого без диагонального сдвига успешно. Уравнение для m=14:

$$
-53.492764u_8-53.492764u_{{13}}+205.985529u_{{14}}-49u_{{15}}-49u_{{20}}=6.003730.
$$

Портрет матрицы: [изображение](lab1_stage_8_1_R1_spy.png). Исходный код: [этап 8.1](lab1_stage_8_1_R1.py).

## 8.2. Якоби, Зейдель и ПВР

На тестовой системе N=32 задано $u_m^*=1+\\sin(2\\pi m/n)+0.1\\cos(6\\pi m/n)$, $b^*=Au^*$. Здесь $\\delta_s=\\|u^*-x^{{(s)}}\\|_2/\\|u^*\\|_2$ измеряет фактическую ошибку решения СЛАУ:

{md_table(["Метод", "Итераций", "Порог η", "Достигнутая ηₛ", "δₛ"], [[METHOD_NAMES[r['method']], str(r['iterations']), f"{float(r['epsilon_res']):.0e}", f"{float(r['eta']):.3e}", f"{float(r['delta']):.3e}"] for r in stage2_test])}

Для R1 на N=64 проверены 11 значений $\\omega$ в [1, 1.95], затем значения в окрестности лучшего с шагом 0.01. Наилучшее **среди проверенных**: $\\omega={sor_value:.2f}$, {int(omega_sor['n_s'])} итераций. Это не теоретический оптимум.

{format_main(stage2_main)}

Исходный код и полный поиск: [этап 8.2](lab1_stage_8_2_R1.py), [протокол](lab1_stage_8_2_R1_results.txt).

## 8.3. CG и PCG

Реализованы CG, PCG с диагональным предобуславливателем и PCG с SSOR. Последний применяется двумя разреженными треугольными решениями без формирования $P^{{-1}}$. Параметр SSOR подобран отдельно от параметра ПВР: среди 1.00, 1.20, 1.40, 1.60, 1.80, 1.90 на N=64 наименьшее число итераций (31) дало $\\omega_{{SSOR}}={ssor_value:.2f}$.

На тестовой системе N=32 оценка обусловленности равна $\\mu_2(A)\\approx{kappa_by_N[32]:.1f}$. Для всех трёх методов достигнута невязка ниже 10⁻⁸, а фактическая ошибка согласуется с оценкой $\\delta_s\\leq\\mu_2(A)\\eta_s$:

{md_table(["Метод", "Итераций", "Порог η", "Достигнутая ηₛ", "δₛ", "μ₂ηₛ"], [[METHOD_NAMES[r['method']], str(r['iterations']), f"{float(r['epsilon_res']):.0e}", f"{float(r['eta']):.3e}", f"{float(r['delta']):.3e}", f"{float(next(x['kappa_eta_bound'] for x in stage3 if x['case']=='test' and x['method']==r['method'])):.3e}"] for r in stage3_test])}

{format_main(stage3_main)}

Полная таблица включает число matvec, время подготовки предобуславливателя, медиану времени итераций после пробного и трёх измерений, достигнутую невязку и память CSR: [протокол 8.3](lab1_stage_8_3_R1_results.txt), [исходный код](lab1_stage_8_3_R1.py).

## 8.4. Независимая проверка

Разреженный прямой решатель и библиотечный CG с rtol=10⁻¹⁰ использованы на N=32, 64, 128. Код возврата библиотечного CG равен нулю во всех случаях. Истинная невязка каждого контрольного решения пересчитана отдельно:

{md_table(["N", "Порог η", "η прямого", "Итераций библиотечного CG", "η библиотечного CG"], [[row[0], "1e-10", *row[1:]] for row in control_rows])}

Относительные расхождения собственных решений с прямым:

{md_table(["N", "Метод", "L₂", "L∞"], error_rows)}

Максимальное расхождение по всем девяти сравнениям: {max_l2:.3e} в L₂, {max_linf:.3e} в L∞. [Код 8.4](lab1_stage_8_4_R1.py) и [полный протокол](lab1_stage_8_4_R1_results.txt).

## 8.5. Сводка, графики и физический результат

[Сводная таблица CSV](lab1_stage_8_5_R1_summary.csv) содержит тестовые и основные запуски этапов 8.2–8.3, а также контрольные решения этапа 8.4. Пустые поля означают показатель, который для данного метода не применяется или не измерялся.

![Истинная нормированная невязка в логарифмическом масштабе](lab1_stage_8_5_R1_residuals.png)

![Число итераций в зависимости от N](lab1_stage_8_5_R1_iterations.png)

![Поле концентрации на сетке N=128](lab1_stage_8_5_R1_field_N128.png)

Полная масса вещества оценена сеточной суммой $M_h=h_xh_y\\sum_{{i,j}}u_{{ij}}$, где граничные значения нулевые. Для решения PCG-SSOR:

{condition_table}

На наиболее подробной сетке $M_h={mass_by_N[128]:.9f}$. Максимальная концентрация {peak_value:.6f} достигается около $(x,y)=({peak_x:.4f},{peak_y:.4f})$, рядом с центром источника. Поглощение $c=1$ и нулевые граничные значения ограничивают концентрацию вдали от источника.

## Выводы

- При удвоении N оценка $\\mu_2(A)$ возрастает примерно в четыре раза: {kappa_by_N[32]:.0f} → {kappa_by_N[64]:.0f} → {kappa_by_N[128]:.0f}. С ростом размера и обусловленности число итераций CG увеличивается с 68 до 141 и 297.
- На N=128 диагональный PCG потребовал 258 итераций, SSOR — 46 вместо 297 для CG. В этой реализации SSOR требует больше времени на итерации из-за двух треугольных решений; сравнивать только число итераций недостаточно.
- Все представленные запуски достигли порога невязки. Устойчивой стагнации и случаев недостижения порога при выбранных параметрах не наблюдалось; в кривых CG возможны временные немонотонные участки.
- Согласие итерационных и контрольных решателей подтверждает решение одной и той же дискретной системы. Точность дискретизации исходного уравнения здесь не оценивалась.

## Запуск и исходный код

Установить NumPy, SciPy и Matplotlib. Из папки проекта запустить последовательно:

```bash
python3 lab1_stage_8_1_R1.py
python3 lab1_stage_8_2_R1.py
python3 lab1_stage_8_3_R1.py
python3 lab1_stage_8_4_R1.py
python3 lab1_stage_8_5_R1.py
```

Все исходники находятся рядом с отчётом: [8.1](lab1_stage_8_1_R1.py), [8.2](lab1_stage_8_2_R1.py), [8.3](lab1_stage_8_3_R1.py), [8.4](lab1_stage_8_4_R1.py), [8.5](lab1_stage_8_5_R1.py). Требования постановки: «Лабораторная работа 1. Итерационные методы решения разреженных систем линейных алгебраических уравнений», разделы 8–9.
"""
    (output_dir / "lab1_R1_report.md").write_text(report, encoding="utf-8")
    print(f"Сводная таблица: {summary_path}")
    print(f"График невязки: {residual_figure}")
    print(f"График итераций: {iterations_figure}")
    print(f"Поле решения: {field_figure}")
    print(f"Отчёт: {output_dir / 'lab1_R1_report.md'}")
    print(f"Масса N=128: {mass_by_N[128]:.9f}; максимум: {peak_value:.6f} при ({peak_x:.4f}, {peak_y:.4f})")
    print("Оценки обусловленности:", ", ".join(f"N={N}: {kappa_by_N[N]:.1f}" for N in (32,64,128)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.project_dir, args.output_dir)


if __name__ == "__main__":
    main()
