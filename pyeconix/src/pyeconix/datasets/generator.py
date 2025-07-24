"""
Data Generator Module for Econometric Simulations

This module provides classes for generating various types of economic and statistical data
for educational purposes in econometrics. It supports different data generating processes,
error structures, and relationship types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, runtime_checkable, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data generation parameters."""

    n_obs: int = 100
    n_vars: int = 1
    intercept: float = 2.0
    coefficients: list[float] = field(default_factory=lambda: [3.0])
    error_variance: float = 1.0
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if len(self.coefficients) != self.n_vars:
            raise ValueError(
                f"Number of coefficients ({len(self.coefficients)}) must match n_vars ({self.n_vars})"
            )


@runtime_checkable
class ErrorDistribution(Protocol):
    """Protocol for error term distributions."""

    def sample(self, size: int, **kwargs: Any) -> NDArray[np.floating]:
        """Generate random samples from the distribution."""
        ...


@dataclass
class NormalError:
    """Normal/Gaussian error distribution."""

    loc: float = 0.0
    scale: float = 1.0

    def sample(
        self, size: int, random_state: Optional[np.random.Generator] = None
    ) -> NDArray[np.floating]:
        rng = random_state or np.random.default_rng()
        return rng.normal(self.loc, self.scale, size)


@dataclass
class StudentTError:
    """Student's t-distribution error (heavy tails)."""

    df: float = 5.0
    loc: float = 0.0
    scale: float = 1.0

    def sample(
        self, size: int, random_state: Optional[np.random.Generator] = None
    ) -> NDArray[np.floating]:
        rng = random_state or np.random.default_rng()
        return np.asarray(stats.t.rvs(
            self.df, loc=self.loc, scale=self.scale, size=size, random_state=rng
        ))


@dataclass
class LaplaceError:
    """Laplace distribution error (double exponential)."""

    loc: float = 0.0
    scale: float = 1.0

    def sample(
        self, size: int, random_state: Optional[np.random.Generator] = None
    ) -> NDArray[np.floating]:
        rng = random_state or np.random.default_rng()
        return rng.laplace(self.loc, self.scale, size)


class BaseDataGenerator(ABC):
    """Abstract base class for all data generators."""

    def __init__(
        self, config: DataConfig, error_dist: Optional[ErrorDistribution] = None
    ) -> None:
        if error_dist is None:
            error_dist = cast(ErrorDistribution, NormalError())
        self.config = config
        self.error_dist = error_dist
        self.rng = np.random.default_rng(config.seed)

    @abstractmethod
    def _generate_features(self) -> NDArray[np.floating]:
        """Generate the feature matrix X."""
        ...

    @abstractmethod
    def _compute_true_y(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute the true (noise-free) dependent variable."""
        ...

    def _generate_errors(self) -> NDArray[np.floating]:
        """Generate error terms."""
        return self.error_dist.sample(
            self.config.n_obs, random_state=self.rng
        ) * np.sqrt(self.config.error_variance)

    def generate(self) -> pd.DataFrame:
        """Generate the complete dataset."""
        X = self._generate_features()
        true_y = self._compute_true_y(X)
        errors = self._generate_errors()
        y = true_y + errors

        # Create column names
        feature_names = [f"x{i + 1}" for i in range(self.config.n_vars)]

        # Build DataFrame
        data = {}
        for i, name in enumerate(feature_names):
            data[name] = X[:, i]

        data.update({"y": y, "true_y": true_y, "error": errors})

        return pd.DataFrame(data)


class LinearDataGenerator(BaseDataGenerator):
    """Generate data with linear relationships."""

    def __init__(
        self,
        config: DataConfig,
        error_dist: Optional[ErrorDistribution] = None,
        x_distribution: Literal["uniform", "normal"] = "uniform",
        x_range: tuple[float, float] = (0.0, 10.0),
    ) -> None:
        super().__init__(config, error_dist)
        self.x_distribution = x_distribution
        self.x_range = x_range

    def _generate_features(self) -> NDArray[np.floating]:
        """Generate feature matrix with specified distribution."""
        if self.x_distribution == "uniform":
            return self.rng.uniform(
                self.x_range[0],
                self.x_range[1],
                (self.config.n_obs, self.config.n_vars),
            )
        elif self.x_distribution == "normal":
            loc = np.mean(self.x_range)
            scale = (
                self.x_range[1] - self.x_range[0]
            ) / 6  # Approximate 99.7% within range
            return self.rng.normal(loc, scale, (self.config.n_obs, self.config.n_vars))
        else:
            raise ValueError(f"Unsupported x_distribution: {self.x_distribution}")

    def _compute_true_y(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute linear relationship: y = intercept + X @ coefficients."""
        return self.config.intercept + X @ np.array(self.config.coefficients)


class NonLinearDataGenerator(BaseDataGenerator):
    """Generate data with non-linear relationships."""

    def __init__(
        self,
        config: DataConfig,
        error_dist: Optional[ErrorDistribution] = None,
        relationship_type: Literal[
            "quadratic", "cubic", "exponential", "logarithmic", "interaction"
        ] = "quadratic",
        x_distribution: Literal["uniform", "normal"] = "uniform",
        x_range: tuple[float, float] = (0.1, 10.0),
    ) -> None:
        super().__init__(config, error_dist)
        self.relationship_type = relationship_type
        self.x_distribution = x_distribution
        self.x_range = x_range

        if config.n_vars > 2 and relationship_type == "interaction":
            raise ValueError(
                "Interaction relationships currently support max 2 variables"
            )

    def _generate_features(self) -> NDArray[np.floating]:
        """Generate feature matrix ensuring positive values for log relationships."""
        if self.x_distribution == "uniform":
            X = self.rng.uniform(
                self.x_range[0],
                self.x_range[1],
                (self.config.n_obs, self.config.n_vars),
            )
        elif self.x_distribution == "normal":
            loc = np.mean(self.x_range)
            scale = (self.x_range[1] - self.x_range[0]) / 6
            X = self.rng.normal(loc, scale, (self.config.n_obs, self.config.n_vars))

            # Ensure positive values for logarithmic relationships
            if self.relationship_type == "logarithmic":
                X = np.abs(X) + 0.1
        else:
            raise ValueError(f"Unsupported x_distribution: {self.x_distribution}")

        return X

    def _compute_true_y(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute non-linear relationship based on specified type."""
        match self.relationship_type:
            case "quadratic":
                return self._quadratic_relationship(X)
            case "cubic":
                return self._cubic_relationship(X)
            case "exponential":
                return self._exponential_relationship(X)
            case "logarithmic":
                return self._logarithmic_relationship(X)
            case "interaction":
                return self._interaction_relationship(X)
            case _:
                raise ValueError(
                    f"Unsupported relationship_type: {self.relationship_type}"
                )

    def _quadratic_relationship(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Quadratic: y = intercept + β₁x₁ + β₂x₁² + ..."""
        linear_part = X @ np.array(self.config.coefficients)
        quadratic_part = 0.5 * np.sum(X**2, axis=1)
        return self.config.intercept + linear_part + quadratic_part

    def _cubic_relationship(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cubic: y = intercept + β₁x₁ + β₂x₁² + β₃x₁³ + ..."""
        linear_part = X @ np.array(self.config.coefficients)
        quadratic_part = 0.5 * np.sum(X**2, axis=1)
        cubic_part = 0.1 * np.sum(X**3, axis=1)
        return self.config.intercept + linear_part + quadratic_part + cubic_part

    def _exponential_relationship(
        self, X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Exponential: y = intercept + exp(β₁x₁ + β₂x₂ + ...)"""
        linear_combination = (
            X @ np.array(self.config.coefficients) * 0.1
        )  # Scale to prevent overflow
        return self.config.intercept + np.exp(linear_combination)

    def _logarithmic_relationship(
        self, X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Logarithmic: y = intercept + β₁log(x₁) + β₂log(x₂) + ..."""
        log_X = np.log(np.maximum(X, 1e-10))  # Prevent log(0)
        return self.config.intercept + log_X @ np.array(self.config.coefficients)

    def _interaction_relationship(
        self, X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Interaction: y = intercept + β₁x₁ + β₂x₂ + β₃x₁x₂"""
        if X.shape[1] < 2:
            raise ValueError("Interaction relationship requires at least 2 variables")

        linear_part = X @ np.array(self.config.coefficients[: X.shape[1]])
        interaction_part = 0.5 * X[:, 0] * X[:, 1]  # x₁ * x₂ interaction
        return self.config.intercept + linear_part + interaction_part


class HeteroskedasticDataGenerator(LinearDataGenerator):
    """Generate data with heteroskedastic errors."""

    def __init__(
        self,
        config: DataConfig,
        error_dist: Optional[ErrorDistribution] = None,
        heteroskedasticity_type: Literal[
            "proportional", "squared", "grouped"
        ] = "proportional",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, error_dist, **kwargs)
        self.heteroskedasticity_type = heteroskedasticity_type

    def _generate_errors(self) -> NDArray[np.floating]:
        """Generate heteroskedastic errors."""
        # First generate features to determine error variance pattern
        X = self._generate_features()

        match self.heteroskedasticity_type:
            case "proportional":
                # Error variance proportional to X
                variance_multiplier = 1 + 0.5 * np.abs(X[:, 0])
            case "squared":
                # Error variance proportional to X²
                variance_multiplier = 1 + 0.1 * X[:, 0] ** 2
            case "grouped":
                # Different error variances for different groups
                median_x = np.median(X[:, 0])
                variance_multiplier = np.where(X[:, 0] > median_x, 2.0, 0.5)
            case _:
                raise ValueError(
                    f"Unsupported heteroskedasticity_type: {self.heteroskedasticity_type}"
                )

        # Generate base errors and scale by variance multiplier
        base_errors = self.error_dist.sample(self.config.n_obs, random_state=self.rng)
        return base_errors * np.sqrt(variance_multiplier * self.config.error_variance)


class AutocorrelatedDataGenerator(LinearDataGenerator):
    """Generate time series data with autocorrelated errors."""

    def __init__(
        self,
        config: DataConfig,
        error_dist: Optional[ErrorDistribution] = None,
        ar_coefficient: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, error_dist, **kwargs)
        self.ar_coefficient = ar_coefficient

        if not -1 < ar_coefficient < 1:
            raise ValueError("AR coefficient must be between -1 and 1 for stationarity")

    def _generate_errors(self) -> NDArray[np.floating]:
        """Generate AR(1) autocorrelated errors: εₜ = ρεₜ₋₁ + uₜ"""
        innovations = self.error_dist.sample(self.config.n_obs, random_state=self.rng)
        innovations *= np.sqrt(self.config.error_variance)

        errors = np.zeros(self.config.n_obs)
        errors[0] = innovations[0]

        for t in range(1, self.config.n_obs):
            errors[t] = self.ar_coefficient * errors[t - 1] + innovations[t]

        return errors


# Factory function for easy generator creation
def create_data_generator(
    generator_type: Literal["linear", "nonlinear", "heteroskedastic", "autocorrelated"],
    config: DataConfig,
    **kwargs: Any,
) -> BaseDataGenerator:
    """Factory function to create data generators."""

    match generator_type:
        case "linear":
            return LinearDataGenerator(config, **kwargs)
        case "nonlinear":
            return NonLinearDataGenerator(config, **kwargs)
        case "heteroskedastic":
            return HeteroskedasticDataGenerator(config, **kwargs)
        case "autocorrelated":
            return AutocorrelatedDataGenerator(config, **kwargs)
        case _:
            raise ValueError(f"Unsupported generator_type: {generator_type}")


if __name__ == "__main__":
    # Example usage
    config = DataConfig(n_obs=200, n_vars=2, coefficients=[2.5, -1.2])

    # Linear data
    linear_gen = create_data_generator("linear", config)
    linear_data = linear_gen.generate()
    print("Linear data sample:")
    print(linear_data.head())

    # Non-linear data
    nonlinear_gen = create_data_generator(
        "nonlinear", config, relationship_type="quadratic"
    )
    nonlinear_data = nonlinear_gen.generate()
    print("\nNon-linear data sample:")
    print(nonlinear_data.head())
