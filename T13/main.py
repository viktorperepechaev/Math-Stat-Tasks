import numpy as np

from scipy.stats import norm

from config import (
    mu,
    sigma_squared,
    number_of_numbers_in_sample,
    number_of_bootstrap_repetitions,
    quantile_levels,
)


def generate_sample(n: int) -> np.ndarray:
    return np.random.normal(
        loc=mu,
        scale=np.sqrt(sigma_squared),
        size=n,
    )


def theoretical_quantile(level: float) -> float:
    return float(norm.ppf(level, loc=mu, scale=np.sqrt(sigma_squared)))


def sample_quantile(sample: np.ndarray, level: float) -> float:
    return float(np.quantile(sample, level))


def bootstrap_quantile_estimates(sample: np.ndarray, level: float) -> np.ndarray:
    n: int = len(sample)
    estimates: np.ndarray = np.zeros(number_of_bootstrap_repetitions, dtype=float)
    for i in range(number_of_bootstrap_repetitions):
        bootstrap_sample: np.ndarray = np.random.choice(sample, size=n, replace=True)
        estimates[i] = sample_quantile(bootstrap_sample, level)
    return estimates


if __name__ == "__main__":
    for level in quantile_levels:
        sample: np.ndarray = generate_sample(number_of_numbers_in_sample)
        theoretical: float = theoretical_quantile(level)
        estimate: float = sample_quantile(sample, level)
        bootstrap_estimates: np.ndarray = bootstrap_quantile_estimates(sample, level)
        bootstrap_std: float = float(np.std(bootstrap_estimates, ddof=1))
        bootstrap_left: float = float(np.quantile(bootstrap_estimates, 0.025))
        bootstrap_right: float = float(np.quantile(bootstrap_estimates, 0.975))

        print(f"Уровень квантили: {level}")
        print(f"Теоретическое T: {theoretical:.2f}")
        print(f"Выборочная оценка T: {estimate:.2f}")
        print(f"Bootstrap-оценка стандартного отклонения: {bootstrap_std:.4f}")
        print(
            f"95%-доверительный интервал бутстрепа: "
            f"[{bootstrap_left:.2f}, {bootstrap_right:.2f}]"
        )
        print()
