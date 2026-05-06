import random
import numpy as np
from scipy.stats import cauchy, rv_continuous
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    left_boundary_theta,
    right_boundary_theta,
    how_many_launches,
    number_of_numbers_in_sample,
    number_of_repetitions,
)


def generate_sample_median_and_mean(static_distribution) -> tuple[float, float]:
    """Second point"""
    sample: np.ndarray = np.asarray(
        static_distribution.rvs(size=number_of_numbers_in_sample)
    )
    with open("log.txt", "a", encoding="utf-16") as file:
        print(f"Первые 12 значений выборки:{sample[:12].tolist()}", file=file)

    """Third point"""
    sample_median: float = np.median(sample)
    sample_mean: float = float(np.mean(sample))
    print(f"Выборочная медиана:{sample_median}", f"Выборочное среднее:{sample_mean}")

    return sample_median, sample_mean


def better_choice(
    distribution: rv_continuous,
    loc_distribution: int | float,
    scale_distribution: int | float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_median: np.ndarray = np.zeros(number_of_repetitions, dtype=float)
    sample_mean: np.ndarray = np.zeros(number_of_repetitions, dtype=float)
    static_distribution = distribution(loc=loc_distribution, scale=scale_distribution)
    """Fourth point"""
    for i in range(number_of_repetitions):
        sample_median[i], sample_mean[i] = generate_sample_median_and_mean(
            static_distribution
        )

    return sample_median, sample_mean


def histograms_comparison(
    theta: float, sample_median: np.ndarray, sample_mean: np.ndarray, launch: int
) -> None:
    canvas: go.Figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Выборочные медианы", f"Выборочные средние")
    )

    width: float = (float(np.max(sample_median)) - float(np.min(sample_median))) / 10

    """Fifth point"""
    canvas.add_trace(
        go.Histogram(name="Выборочные медианы", x=sample_median, xbins=dict(size=width)), row=1, col=1
    )
    canvas.add_vline(
        x=theta, line_dash="dash", line_color="red", annotation_text="θ", row=1, col=1
    )

    """Sixth point"""
    canvas.add_trace(
        go.Histogram(name="Выборочные средние", x=sample_mean, xbins=dict(size=width)), row=1, col=2
    )
    canvas.add_vline(
        x=theta, line_dash="dash", line_color="red", annotation_text="θ", row=1, col=2
    )

    canvas.update_layout(
        title_text=f"Сравнение оценок. Запуск №: {launch}. θ: {theta}"
    )

    canvas.write_html(f"Запуск №: {launch}.html")


if __name__ == "__main__":
    with open("log.txt", "w", encoding="utf-16") as f:
        print("--------------------", file=f)

    """Eight point"""
    for i in range(how_many_launches):
        print(f"Запуск №{i + 1}")
        """First point"""
        theta: float = random.uniform(left_boundary_theta, right_boundary_theta)
        print(f"θ:{theta}")

        sample_median, sample_mean = better_choice(
            distribution=cauchy, loc_distribution=theta, scale_distribution=1
        )

        histograms_comparison(theta, sample_median, sample_mean, i + 1)

    with open("log.txt", "a", encoding="utf-16") as f:
        print("--------------------", file=f)
