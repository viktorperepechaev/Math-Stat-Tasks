import numpy as np

from scipy.stats import norm

import plotly.graph_objects as go

from config import (
    number_of_numbers_in_sample,
    number_of_repetitions,
    left_boundary_x,
    right_boundary_x,
    number_of_points_on_graph,
)


def generate_sample(n: int) -> np.ndarray:
    return np.random.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )


def kolmogorov_statistic(sample: np.ndarray) -> float:
    n: int = len(sample)
    sorted_sample: np.ndarray = np.sort(sample)
    theoretical_values: np.ndarray = norm.cdf(sorted_sample)
    numbers: np.ndarray = np.arange(1, n + 1)
    right_differences: np.ndarray = numbers / n - theoretical_values
    left_differences: np.ndarray = theoretical_values - (numbers - 1) / n
    return float(np.max(np.maximum(right_differences, left_differences)))


def calculate_statistics_for_prefixes(sample: np.ndarray) -> np.ndarray:
    statistics: np.ndarray = np.zeros(len(sample), dtype=float)
    for n in range(1, len(sample) + 1):
        statistics[n - 1] = kolmogorov_statistic(sample[:n])
    return statistics


def graphic_statistics(statistics: np.ndarray) -> None:
    n_values: np.ndarray = np.arange(1, len(statistics) + 1)
    canvas: go.Figure = go.Figure()
    canvas.add_trace(
        go.Scatter(
            x=n_values,
            y=statistics,
            mode="lines",
            name="Dn",
        )
    )
    canvas.update_layout(
        title="Зависимость Dn от n",
        xaxis_title="n",
        yaxis_title="Dn",
    )
    canvas.write_html("T10_D_n.html")


def calculate_sqrt_statistics() -> np.ndarray:
    sqrt_statistics: np.ndarray = np.zeros(number_of_repetitions, dtype=float)
    for i in range(number_of_repetitions):
        sample: np.ndarray = generate_sample(number_of_numbers_in_sample)
        statistics: float = kolmogorov_statistic(sample)
        sqrt_statistics[i] = np.sqrt(number_of_numbers_in_sample) * statistics
    return sqrt_statistics


def sqrt_distribution(sqrt_statistics: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    used: np.ndarray = np.zeros(len(x_values), dtype=float)
    for i in range(len(x_values)):
        used[i] = np.mean(sqrt_statistics <= x_values[i])
    return used


def kolmogorov_distribution(x_values: np.ndarray) -> np.ndarray:
    values: np.ndarray = np.zeros(len(x_values), dtype=float)
    for i, x in enumerate(x_values):
        if x <= 0:
            values[i] = 0.0
        else:
            terms: np.ndarray = np.array(
                [(-1) ** (k - 1) * np.exp(-2.0 * k**2 * x**2) for k in range(1, 101)]
            )
            values[i] = 1.0 - 2.0 * float(np.sum(terms))
    return values


def graphic_distribution(
    x_values: np.ndarray,
    empirical_values: np.ndarray,
    theoretical_values: np.ndarray,
) -> None:
    canvas: go.Figure = go.Figure()
    canvas.add_trace(
        go.Scatter(
            x=x_values,
            y=empirical_values,
            mode="lines",
            name="Приближённая P(√nDₙ ≤ x)",
        )
    )
    canvas.add_trace(
        go.Scatter(
            x=x_values,
            y=theoretical_values,
            mode="lines",
            name="K(x)",
        )
    )
    canvas.update_layout(
        title="Сравнение эмпирической функции распределения и K(x)",
        xaxis_title="x",
        yaxis_title="Значение функции распределения",
    )
    canvas.write_html("T10_distribution.html")


if __name__ == "__main__":
    sample: np.ndarray = generate_sample(number_of_numbers_in_sample)
    statistics: np.ndarray = calculate_statistics_for_prefixes(sample)
    graphic_statistics(statistics)
    print(f"D_1000:{statistics[-1]}")
    sqrt_statistics: np.ndarray = calculate_sqrt_statistics()
    x_values: np.ndarray = np.linspace(
        left_boundary_x,
        right_boundary_x,
        number_of_points_on_graph,
    )
    empirical_values: np.ndarray = sqrt_distribution(
        sqrt_statistics,
        x_values,
    )
    theoretical_values: np.ndarray = kolmogorov_distribution(x_values)
    graphic_distribution(
        x_values,
        empirical_values,
        theoretical_values,
    )
