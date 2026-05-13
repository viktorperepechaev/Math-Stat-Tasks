import numpy as np
from scipy.stats import f
import plotly.graph_objects as go

from config import (
    theta_1,
    theta_2,
    sigma_squared,
    max_number_of_experiments,
    alpha,
)


def generate_experiment(n: int) -> tuple[np.ndarray, np.ndarray]:
    x_values: np.ndarray = np.random.normal(
        loc=theta_1,
        scale=np.sqrt(sigma_squared),
        size=n,
    )

    y_values: np.ndarray = np.random.normal(
        loc=theta_2,
        scale=np.sqrt(sigma_squared),
        size=2 * n,
    )

    z_values: np.ndarray = np.random.normal(
        loc=theta_1 + theta_2,
        scale=np.sqrt(sigma_squared),
        size=n,
    )

    observations: np.ndarray = np.concatenate((x_values, y_values, z_values))

    design_matrix: np.ndarray = np.vstack(
        (
            np.tile([1.0, 0.0], (n, 1)),
            np.tile([0.0, 1.0], (2 * n, 1)),
            np.tile([1.0, 1.0], (n, 1)),
        )
    )

    return design_matrix, observations


def unrestricted_estimation(
    design_matrix: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, float]:
    estimation: np.ndarray = np.linalg.inv(design_matrix.T @ design_matrix) @ (
        design_matrix.T @ observations
    )

    residuals: np.ndarray = observations - design_matrix @ estimation
    residual_sum: float = float(residuals.T @ residuals)

    return estimation, residual_sum


def restricted_estimation(
    design_matrix: np.ndarray,
    observations: np.ndarray,
    restriction: np.ndarray,
) -> float:
    unrestricted, _ = unrestricted_estimation(design_matrix, observations)

    inverse_matrix: np.ndarray = np.linalg.inv(design_matrix.T @ design_matrix)

    correction: np.ndarray = (
        inverse_matrix
        @ restriction.T
        @ np.linalg.inv(restriction @ inverse_matrix @ restriction.T)
        @ (restriction @ unrestricted)
    )

    restricted: np.ndarray = unrestricted - correction

    residuals: np.ndarray = observations - design_matrix @ restricted
    residual_sum: float = float(residuals.T @ residuals)

    return residual_sum


def f_test(
    design_matrix: np.ndarray,
    observations: np.ndarray,
    restriction: np.ndarray,
) -> bool:
    _, unrestricted_residual_sum = unrestricted_estimation(
        design_matrix,
        observations,
    )

    restricted_residual_sum: float = restricted_estimation(
        design_matrix,
        observations,
        restriction,
    )

    number_of_observations: int = len(observations)
    number_of_parameters: int = design_matrix.shape[1]
    number_of_restrictions: int = restriction.shape[0]

    statistic: float = (
        (restricted_residual_sum - unrestricted_residual_sum)
        / number_of_restrictions
    ) / (
        unrestricted_residual_sum
        / (number_of_observations - number_of_parameters)
    )

    critical_value: float = float(
        f.ppf(
            1.0 - alpha,
            number_of_restrictions,
            number_of_observations - number_of_parameters,
        )
    )

    return statistic > critical_value


def correct_answers_for_hypothesis(
    restriction: np.ndarray,
    true_hypothesis: bool,
) -> np.ndarray:
    correct_answers: np.ndarray = np.zeros(max_number_of_experiments, dtype=float)

    for n in range(1, max_number_of_experiments + 1):
        design_matrix, observations = generate_experiment(n)
        is_rejected: bool = f_test(design_matrix, observations, restriction)

        if true_hypothesis:
            correct_answers[n - 1] = float(not is_rejected)
        else:
            correct_answers[n - 1] = float(is_rejected)

    return correct_answers


def cumulative_correct_answers(correct_answers: np.ndarray) -> np.ndarray:
    return np.cumsum(correct_answers) / np.arange(1, len(correct_answers) + 1)


def graphic_comparison(
    first_correct_answers: np.ndarray,
    second_correct_answers: np.ndarray,
) -> None:
    x_values: np.ndarray = np.arange(1, max_number_of_experiments + 1)

    first_cumulative: np.ndarray = cumulative_correct_answers(first_correct_answers)
    second_cumulative: np.ndarray = cumulative_correct_answers(second_correct_answers)

    canvas: go.Figure = go.Figure()

    canvas.add_trace(
        go.Scatter(
            x=x_values,
            y=first_cumulative,
            mode="lines",
            name="H₀: 8θ₁ = 5θ₂",
        )
    )

    canvas.add_trace(
        go.Scatter(
            x=x_values,
            y=second_cumulative,
            mode="lines",
            name="H₀: 2θ₁ = 3θ₂",
        )
    )

    canvas.update_layout(
        title="Доля правильных ответов в первых n экспериментах",
        xaxis_title="n",
        yaxis_title="Доля правильных ответов",
    )

    canvas.write_html("T9_correct_answers.html")


def optimal_estimations_comparison(n: int) -> None:
    design_matrix, observations = generate_experiment(n)

    optimal, _ = unrestricted_estimation(design_matrix, observations)

    x_values: np.ndarray = observations[:n]
    y_values: np.ndarray = observations[n : 3 * n]

    simple_theta_1: float = float(np.mean(x_values))
    simple_theta_2: float = float(np.mean(y_values))

    print(f"Оптимальная оценка theta_1:{optimal[0]}")
    print(f"Оценка theta_1 только по X:{simple_theta_1}")
    print(f"Оптимальная оценка theta_2:{optimal[1]}")
    print(f"Оценка theta_2 только по Y:{simple_theta_2}")


if __name__ == "__main__":
    first_restriction: np.ndarray = np.array([[8.0, -5.0]])
    second_restriction: np.ndarray = np.array([[2.0, -3.0]])

    first_correct_answers: np.ndarray = correct_answers_for_hypothesis(
        first_restriction,
        true_hypothesis=True,
    )

    second_correct_answers: np.ndarray = correct_answers_for_hypothesis(
        second_restriction,
        true_hypothesis=False,
    )

    graphic_comparison(first_correct_answers, second_correct_answers)

    optimal_estimations_comparison(max_number_of_experiments)