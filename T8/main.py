import random
import numpy as np
from scipy.stats import uniform
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    left_boundary_theta,
    right_boundary_theta,
    how_many_launches,
    number_of_numbers_in_sample,
    number_of_repetitions,
)


def generate_sample_optimal_and_moment(static_distribution) -> tuple[float, float]:
    sample: np.ndarray = np.asarray(
        static_distribution.rvs(size=number_of_numbers_in_sample)
    )

    with open("log.txt", "a", encoding="utf-16") as file:
        print(f"Первые 12 значений выборки:{sample[:12].tolist()}", file=file)

    optimal_estimation: float = (
        (number_of_numbers_in_sample + 1)
        / number_of_numbers_in_sample
        * float(np.max(sample))
    )

    moment_estimation: float = 2.0 * float(np.mean(sample))

    print(
        f"Оптимальная оценка:{optimal_estimation}",
        f"Оценка методом моментов:{moment_estimation}",
    )

    return optimal_estimation, moment_estimation


def estimations_comparison(theta: float) -> tuple[np.ndarray, np.ndarray]:
    optimal_estimations: np.ndarray = np.zeros(number_of_repetitions, dtype=float)
    moment_estimations: np.ndarray = np.zeros(number_of_repetitions, dtype=float)

    static_distribution = uniform(loc=0.0, scale=theta)

    for i in range(number_of_repetitions):
        optimal_estimations[i], moment_estimations[i] = (
            generate_sample_optimal_and_moment(static_distribution)
        )

    return optimal_estimations, moment_estimations


def histograms_comparison(
    theta: float,
    optimal_estimations: np.ndarray,
    moment_estimations: np.ndarray,
    launch: int,
    prediction_number: int,
) -> None:
    canvas: go.Figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Оптимальные оценки",
            "Оценки методом моментов",
        ),
    )

    left: float = float(
        min(np.min(optimal_estimations), np.min(moment_estimations), theta)
    )
    right: float = float(
        max(np.max(optimal_estimations), np.max(moment_estimations), theta)
    )
    width: float = (right - left) / 20

    canvas.add_trace(
        go.Histogram(
            name="Оптимальные оценки",
            x=optimal_estimations,
            xbins=dict(size=width),
        ),
        row=1,
        col=1,
    )

    canvas.add_vline(
        x=theta,
        line_dash="dash",
        line_color="red",
        annotation_text="θ",
        row=1,
        col=1,
    )

    canvas.add_trace(
        go.Histogram(
            name="Оценки методом моментов",
            x=moment_estimations,
            xbins=dict(size=width),
        ),
        row=1,
        col=2,
    )

    canvas.add_vline(
        x=theta,
        line_dash="dash",
        line_color="red",
        annotation_text="θ",
        row=1,
        col=2,
    )

    canvas.update_layout(
        title_text=f"Сравнение оценок. Запуск №: {launch}.{prediction_number} θ: {theta}"
    )

    canvas.write_html(f"Запуск №: {launch}_{prediction_number}.html")


def variance_comparison(
    theta: float,
    optimal_estimations: np.ndarray,
    moment_estimations: np.ndarray,
) -> None:
    n: int = number_of_numbers_in_sample

    optimal_variance_theory: float = theta**2 / (n * (n + 2))
    moment_variance_theory: float = theta**2 / (3 * n)
    cramer_rao_regular_value: float = theta**2 / n

    optimal_variance_numeric: float = float(np.var(optimal_estimations, ddof=1))
    moment_variance_numeric: float = float(np.var(moment_estimations, ddof=1))

    print(f"Теоретическая дисперсия оптимальной оценки:{optimal_variance_theory}")
    print(f"Численная дисперсия оптимальной оценки:{optimal_variance_numeric}")
    print(f"Теоретическая дисперсия оценки методом моментов:{moment_variance_theory}")
    print(f"Численная дисперсия оценки методом моментов:{moment_variance_numeric}")
    print(f"Формальное значение 1 / (n i(θ)):{cramer_rao_regular_value}")

    with open("log.txt", "a", encoding="utf-16") as file:
        print(f"θ:{theta}", file=file)
        print(
            f"Теоретическая дисперсия оптимальной оценки:{optimal_variance_theory}",
            file=file,
        )
        print(
            f"Численная дисперсия оптимальной оценки:{optimal_variance_numeric}",
            file=file,
        )
        print(
            f"Теоретическая дисперсия оценки методом моментов:{moment_variance_theory}",
            file=file,
        )
        print(
            f"Численная дисперсия оценки методом моментов:{moment_variance_numeric}",
            file=file,
        )
        print(f"Формальное значение 1 / (n i(θ)):{cramer_rao_regular_value}", file=file)


if __name__ == "__main__":
    with open("log.txt", "w", encoding="utf-16") as f:
        print("--------------------", file=f)

    for i in range(how_many_launches):
        print(f"Запуск №{i + 1}")

        theta: float = random.uniform(left_boundary_theta, right_boundary_theta)
        print(f"θ:{theta}")

        optimal_estimations, moment_estimations = estimations_comparison(theta)

        histograms_comparison(
            theta,
            optimal_estimations,
            moment_estimations,
            i + 1,
            1,
        )

        variance_comparison(
            theta,
            optimal_estimations,
            moment_estimations,
        )

        histograms_comparison(
            0.0,
            number_of_numbers_in_sample * (optimal_estimations - theta),
            np.sqrt(number_of_numbers_in_sample) * (moment_estimations - theta),
            i + 1,
            2,
        )

    with open("log.txt", "a", encoding="utf-16") as f:
        print("--------------------", file=f)
