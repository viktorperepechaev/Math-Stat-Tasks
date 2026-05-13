import numpy as np
from scipy.stats import chi2
import plotly.graph_objects as go

from config import (
    left_boundary_theta,
    right_boundary_theta,
    number_of_numbers_in_sample,
    alpha,
)


def generate_theta_values() -> np.ndarray:
    return np.arange(
        int(10 * left_boundary_theta),
        int(10 * right_boundary_theta) + 1,
        1,
        dtype=float,
    ) / 10


def generate_sample(theta: float, n: int) -> np.ndarray:
    return np.random.poisson(
        lam=theta,
        size=n,
    )


def exact_confidence_interval(sample: np.ndarray) -> tuple[float, float]:
    n: int = len(sample)
    sample_sum: int = int(np.sum(sample))

    if sample_sum == 0:
        left: float = 0.0
    else:
        left = float(chi2.ppf(alpha / 2.0, 2 * sample_sum) / (2 * n))

    right: float = float(chi2.ppf(1.0 - alpha / 2.0, 2 * sample_sum + 2) / (2 * n))

    return left, right


def calculate_intervals(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_values: np.ndarray = generate_theta_values()
    left_values: np.ndarray = np.zeros(len(theta_values), dtype=float)
    right_values: np.ndarray = np.zeros(len(theta_values), dtype=float)

    for i, theta in enumerate(theta_values):
        sample: np.ndarray = generate_sample(theta, n)
        left_values[i], right_values[i] = exact_confidence_interval(sample)

    return theta_values, left_values, right_values


def graphic_intervals(
    theta_values: np.ndarray,
    left_values: np.ndarray,
    right_values: np.ndarray,
    n: int,
) -> None:
    canvas: go.Figure = go.Figure()

    for theta, left, right in zip(theta_values, left_values, right_values):
        canvas.add_trace(
            go.Scatter(
                x=[theta, theta],
                y=[left, right],
                mode="lines",
                showlegend=False,
            )
        )

    canvas.add_trace(
        go.Scatter(
            x=theta_values,
            y=theta_values,
            mode="lines",
            name="f(θ) = θ",
        )
    )

    canvas.update_layout(
        title=f"Точные доверительные интервалы для θ при n = {n}",
        xaxis_title="Истинное значение θ",
        yaxis_title="Доверительный интервал",
    )

    canvas.write_html(f"T12_n_{n}.html")


if __name__ == "__main__":
    for n in [number_of_numbers_in_sample, 5, 10, 100]:
        theta_values, left_values, right_values = calculate_intervals(n)

        graphic_intervals(
            theta_values,
            left_values,
            right_values,
            n,
        )