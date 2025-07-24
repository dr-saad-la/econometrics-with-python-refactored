"""
Examples for Data Generator Module

This script demonstrates how to use the various data generators
for creating synthetic datasets for econometric analysis.
"""

import sys
from pathlib import Path
import pandas as pd


# Add the src directory to path so we can import pyeconix
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pyeconix.datasets.generator import (
    DataConfig,
    create_data_generator,
    NormalError,
    StudentTError,
    LaplaceError,
)


def demonstrate_basic_generators():
    """Demonstrate basic data generator types."""
    print("=" * 60)
    print("BASIC DATA GENERATORS DEMONSTRATION")
    print("=" * 60)

    # Configuration for all examples
    config = DataConfig(n_obs=100, n_vars=2, coefficients=[2.5, -1.2], seed=42)

    print(f"Configuration: {config}")
    print(
        f"True relationship: y = {config.intercept} + {config.coefficients[0]}*x1 + {config.coefficients[1]}*x2 + error\n"
    )

    # 1. Linear Data Generator
    print("1. LINEAR DATA GENERATOR")
    print("-" * 30)

    linear_gen = create_data_generator("linear", config)
    linear_data = linear_gen.generate()

    print("Sample of linear data:")
    print(linear_data.head())
    print(f"\nData shape: {linear_data.shape}")
    print(f"Columns: {list(linear_data.columns)}")

    # Show correlation between true_y and observed y
    correlation = linear_data["y"].corr(linear_data["true_y"])
    print(f"Correlation between true_y and observed y: {correlation:.3f}")

    # 2. Non-linear Data Generator
    print("\n2. NON-LINEAR DATA GENERATOR (Quadratic)")
    print("-" * 45)

    nonlinear_gen = create_data_generator(
        "nonlinear", config, relationship_type="quadratic"
    )
    nonlinear_data = nonlinear_gen.generate()

    print("Sample of non-linear data:")
    print(nonlinear_data.head())

    # Compare variance in true_y
    linear_var = linear_data["true_y"].var()
    nonlinear_var = nonlinear_data["true_y"].var()
    print(
        f"\nVariance in true_y - Linear: {linear_var:.2f}, Non-linear: {nonlinear_var:.2f}"
    )


def demonstrate_error_distributions():
    """Demonstrate different error distributions."""
    print("\n\n" + "=" * 60)
    print("ERROR DISTRIBUTIONS DEMONSTRATION")
    print("=" * 60)

    config = DataConfig(n_obs=200, coefficients=[3.0], error_variance=2.0, seed=123)

    # Different error distributions
    error_distributions = {
        "Normal": NormalError(loc=0.0, scale=1.0),
        "Student-t (df=3)": StudentTError(df=3.0),
        "Student-t (df=10)": StudentTError(df=10.0),
        "Laplace": LaplaceError(loc=0.0, scale=1.0),
    }

    results = {}

    for name, error_dist in error_distributions.items():
        print(f"\n{name} Error Distribution:")
        print("-" * (len(name) + 20))

        gen = create_data_generator("linear", config, error_dist=error_dist)
        data = gen.generate()

        # Calculate error statistics
        errors = data["error"]
        results[name] = {
            "mean": errors.mean(),
            "std": errors.std(),
            "skewness": errors.skew(),
            "kurtosis": errors.kurtosis(),
            "min": errors.min(),
            "max": errors.max(),
        }

        print("Error statistics:")
        print(f"  Mean: {results[name]['mean']:.3f}")
        print(f"  Std:  {results[name]['std']:.3f}")
        print(f"  Skew: {results[name]['skewness']:.3f}")
        print(f"  Kurt: {results[name]['kurtosis']:.3f}")
        print(f"  Range: [{results[name]['min']:.2f}, {results[name]['max']:.2f}]")

    # Summary comparison
    print(f"\n{'Distribution':<15} {'Mean':<8} {'Std':<8} {'Skew':<8} {'Kurt':<8}")
    print("-" * 55)
    for name, stats in results.items():
        print(
            f"{name:<15} {stats['mean']:<8.3f} {stats['std']:<8.3f} {stats['skewness']:<8.3f} {stats['kurtosis']:<8.3f}"
        )


def demonstrate_nonlinear_relationships():
    """Demonstrate different non-linear relationship types."""
    print("\n\n" + "=" * 60)
    print("NON-LINEAR RELATIONSHIPS DEMONSTRATION")
    print("=" * 60)

    config = DataConfig(n_obs=150, coefficients=[1.5], error_variance=1.0, seed=456)

    relationship_types = ["quadratic", "cubic", "exponential", "logarithmic"]

    for rel_type in relationship_types:
        print(f"\n{rel_type.upper()} RELATIONSHIP:")
        print("-" * (len(rel_type) + 15))

        try:
            gen = create_data_generator(
                "nonlinear",
                config,
                relationship_type=rel_type,
                x_range=(0.5, 5.0),  # Ensure positive values for log
            )
            data = gen.generate()

            print("Sample data:")
            print(data[["x1", "true_y", "y"]].head())

            # Show range of true_y values
            true_y_range = data["true_y"].max() - data["true_y"].min()
            print(f"Range of true_y: {true_y_range:.2f}")

        except Exception as e:
            print(f"Error generating {rel_type} data: {e}")


def demonstrate_interaction_effects():
    """Demonstrate interaction effects with multiple variables."""
    print("\n\n" + "=" * 60)
    print("INTERACTION EFFECTS DEMONSTRATION")
    print("=" * 60)

    config = DataConfig(
        n_obs=100, n_vars=2, coefficients=[2.0, 1.5], error_variance=0.5, seed=789
    )

    print("Generating data with interaction effects...")
    print("Model: y = intercept + β₁x₁ + β₂x₂ + β₃x₁x₂ + error")

    gen = create_data_generator("nonlinear", config, relationship_type="interaction")
    data = gen.generate()

    print("\nSample data:")
    print(data.head())

    # Calculate the interaction term manually for verification
    data["x1_x2_interaction"] = data["x1"] * data["x2"]

    print("\nCorrelations:")
    correlations = data[["x1", "x2", "x1_x2_interaction", "y"]].corr()["y"].drop("y")
    for var, corr in correlations.items():
        print(f"  {var} with y: {corr:.3f}")


def demonstrate_heteroskedasticity():
    """Demonstrate heteroskedastic data generation."""
    print("\n\n" + "=" * 60)
    print("HETEROSKEDASTICITY DEMONSTRATION")
    print("=" * 60)

    config = DataConfig(n_obs=200, coefficients=[2.0], error_variance=1.0, seed=101)

    hetero_types = ["proportional", "squared", "grouped"]

    for hetero_type in hetero_types:
        print(f"\n{hetero_type.upper()} HETEROSKEDASTICITY:")
        print("-" * (len(hetero_type) + 20))

        gen = create_data_generator(
            "heteroskedastic", config, heteroskedasticity_type=hetero_type
        )
        data = gen.generate()

        # Analyze error variance patterns
        data["x1_groups"] = pd.cut(
            data["x1"], bins=5, labels=["Low", "Low-Med", "Med", "Med-High", "High"]
        )
        error_by_group = data.groupby("x1_groups")["error"].agg(["mean", "std", "var"])

        print("Error variance by x1 groups:")
        print(error_by_group)

        # Show variance ratio
        var_ratio = error_by_group["var"].max() / error_by_group["var"].min()
        print(f"Variance ratio (max/min): {var_ratio:.2f}")


def demonstrate_autocorrelation():
    """Demonstrate autocorrelated time series data."""
    print("\n\n" + "=" * 60)
    print("AUTOCORRELATION DEMONSTRATION")
    print("=" * 60)

    config = DataConfig(n_obs=100, coefficients=[1.5], error_variance=1.0, seed=202)

    ar_coefficients = [0.0, 0.3, 0.7, 0.9]

    for ar_coef in ar_coefficients:
        print(f"\nAR COEFFICIENT: {ar_coef}")
        print("-" * 25)

        if ar_coef == 0.0:
            # No autocorrelation case
            gen = create_data_generator("linear", config)
        else:
            gen = create_data_generator(
                "autocorrelated", config, ar_coefficient=ar_coef
            )

        data = gen.generate()

        # Calculate autocorrelation
        error_autocorr = data["error"].autocorr(lag=1)

        print(f"First-order autocorrelation of errors: {error_autocorr:.3f}")
        print(f"Error variance: {data['error'].var():.3f}")

        # Show first few error terms
        print("First 10 error terms:")
        print(data["error"].head(10).round(3).tolist())


def demonstrate_custom_configurations():
    """Demonstrate custom configurations and advanced usage."""
    print("\n\n" + "=" * 60)
    print("CUSTOM CONFIGURATIONS DEMONSTRATION")
    print("=" * 60)

    # Complex configuration
    custom_config = DataConfig(
        n_obs=300,
        n_vars=3,
        intercept=5.0,
        coefficients=[2.5, -1.8, 0.7],
        error_variance=4.0,
        seed=303,
    )

    print("Custom configuration:")
    print(f"  Observations: {custom_config.n_obs}")
    print(f"  Variables: {custom_config.n_vars}")
    print(f"  Intercept: {custom_config.intercept}")
    print(f"  Coefficients: {custom_config.coefficients}")
    print(f"  Error variance: {custom_config.error_variance}")

    # Generate with custom error distribution
    heavy_tail_error = StudentTError(df=2.5, scale=1.5)

    gen = create_data_generator(
        "linear",
        custom_config,
        error_dist=heavy_tail_error,
        x_distribution="normal",
        x_range=(-2.0, 8.0),
    )

    data = gen.generate()

    print(f"\nGenerated data shape: {data.shape}")
    print("\nSummary statistics:")
    print(data.describe().round(3))

    # Calculate R-squared
    ss_res = ((data["y"] - data["true_y"]) ** 2).sum()
    ss_tot = ((data["y"] - data["y"].mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)
    print(f"\nR-squared: {r_squared:.3f}")


def main():
    """Run all demonstrations."""
    print("PYECONIX DATA GENERATORS - COMPREHENSIVE EXAMPLES")
    print("=" * 60)

    try:
        demonstrate_basic_generators()
        demonstrate_error_distributions()
        demonstrate_nonlinear_relationships()
        demonstrate_interaction_effects()
        demonstrate_heteroskedasticity()
        demonstrate_autocorrelation()
        demonstrate_custom_configurations()

        print("\n\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except ImportError as e:
        print(f"Import error: {e}")
        print(
            "Make sure the pyeconix package is properly installed or in the Python path."
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
