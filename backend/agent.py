"""
Basis-Selection Agent
----------------------
Given noisy (x, y) observations, fits several candidate basis families
(B-spline, Fourier, Gaussian RBF), scores each via GCV, selects the best,
and generates a natural-language rationale from residual diagnostics.
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lstsq


# ---------- Fitting ----------

def _fit_and_score(X_design, y):
    n, k = X_design.shape
    coefs, *_ = lstsq(X_design, y)
    y_hat = X_design @ coefs

    XtX_inv = np.linalg.pinv(X_design.T @ X_design)
    H = X_design @ XtX_inv @ X_design.T
    trace_H = np.trace(H)

    rss = np.sum((y - y_hat) ** 2)
    if n <= trace_H:
        raise ValueError(
            f"Degrees of freedom exhausted (n={n}, trace(H)={trace_H:.2f}); "
            "cannot compute a finite GCV score for this basis."
        )
    gcv = (n * rss) / (n - trace_H) ** 2

    if not np.isfinite(gcv):
        raise ValueError("GCV score is not finite for this basis configuration.")

    return coefs, y_hat, gcv


def fit_bspline(x, y, n_knots=10, degree=3):
    knots = np.linspace(x.min(), x.max(), n_knots)
    knots = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    n_basis = len(knots) - degree - 1

    if n_basis >= len(x):
        raise ValueError(
            f"B-spline with n_knots={n_knots} needs {n_basis} basis functions, "
            f"but only {len(x)} observations were provided."
        )

    X_design = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        coef = np.zeros(n_basis)
        coef[i] = 1
        spline = BSpline(knots, coef, degree)
        X_design[:, i] = spline(x)

    _, y_hat, gcv = _fit_and_score(X_design, y)
    return {"name": "B-spline", "params": {"n_knots": n_knots}, "y_hat": y_hat, "gcv": gcv, "n_basis": n_basis}


def fit_fourier(x, y, n_harmonics=8):
    n_basis = 2 * n_harmonics + 1
    if n_basis >= len(x):
        raise ValueError(
            f"Fourier with n_harmonics={n_harmonics} needs {n_basis} basis functions, "
            f"but only {len(x)} observations were provided."
        )
    X_design = np.ones((len(x), n_basis))
    for k in range(1, n_harmonics + 1):
        X_design[:, 2 * k - 1] = np.sin(2 * np.pi * k * x)
        X_design[:, 2 * k] = np.cos(2 * np.pi * k * x)

    _, y_hat, gcv = _fit_and_score(X_design, y)
    return {"name": "Fourier", "params": {"n_harmonics": n_harmonics}, "y_hat": y_hat, "gcv": gcv, "n_basis": X_design.shape[1]}


def fit_gaussian_rbf(x, y, n_centers=10, bandwidth=0.1):
    n_basis = n_centers + 1
    if n_basis >= len(x):
        raise ValueError(
            f"Gaussian RBF with n_centers={n_centers} needs {n_basis} basis functions, "
            f"but only {len(x)} observations were provided."
        )
    centers = np.linspace(x.min(), x.max(), n_centers)
    X_design = np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * bandwidth ** 2))
    X_design = np.c_[np.ones(len(x)), X_design]

    _, y_hat, gcv = _fit_and_score(X_design, y)
    return {"name": "Gaussian RBF", "params": {"bandwidth": bandwidth}, "y_hat": y_hat, "gcv": gcv, "n_basis": X_design.shape[1]}


# ---------- Selection ----------

def select_best_basis(x, y):
    candidates = []

    for n_knots in [5, 10, 15]:
        try:
            candidates.append(fit_bspline(x, y, n_knots=n_knots))
        except Exception:
            continue

    for n_harmonics in [3, 6, 10]:
        try:
            candidates.append(fit_fourier(x, y, n_harmonics=n_harmonics))
        except Exception:
            continue

    for bandwidth in [0.05, 0.1, 0.2]:
        try:
            candidates.append(fit_gaussian_rbf(x, y, bandwidth=bandwidth))
        except Exception:
            continue

    if not candidates:
        raise ValueError(
            "No basis family could be fit with the given number of observations. "
            "Try providing more data points (at least ~20 is recommended)."
        )

    best = min(candidates, key=lambda c: c["gcv"])

    best_per_family = {}
    for c in candidates:
        fam = c["name"]
        if fam not in best_per_family or c["gcv"] < best_per_family[fam]["gcv"]:
            best_per_family[fam] = c

    return {"best": best, "all_candidates": candidates, "best_per_family": best_per_family}


# ---------- Rationale ----------

def _residual_diagnostics(x, y, y_hat):
    resid = y - y_hat
    resid_centered = resid - resid.mean()
    lag1_autocorr = (
        np.sum(resid_centered[:-1] * resid_centered[1:]) / np.sum(resid_centered ** 2)
    ) if np.sum(resid_centered ** 2) > 0 else 0.0

    dy = np.diff(y_hat) / np.diff(x)
    curvature = np.abs(np.diff(dy))
    max_curvature_loc = float(x[1:-1][np.argmax(curvature)]) if len(curvature) > 0 else None

    sign_changes = int(np.sum(np.diff(np.sign(dy)) != 0))

    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "lag1_autocorr": float(lag1_autocorr),
        "max_curvature_loc": max_curvature_loc,
        "oscillations": sign_changes,
    }


def generate_rationale(x, y, result, x_original=None):
    """
    x: normalized x (used for fitting/diagnostics math)
    x_original: original-scale x, used only for human-readable messages
    """
    best = result["best"]
    per_family = result["best_per_family"]
    diag = _residual_diagnostics(x, y, best["y_hat"])

    # Map the normalized curvature location back to the original x scale for display
    if diag["max_curvature_loc"] is not None and x_original is not None:
        idx = np.argmin(np.abs(x - diag["max_curvature_loc"]))
        diag["max_curvature_loc_display"] = float(x_original[idx])
    else:
        diag["max_curvature_loc_display"] = diag["max_curvature_loc"]

    others = [c for c in per_family.values() if c["name"] != best["name"]]
    runner_up = min(others, key=lambda c: c["gcv"]) if others else None

    lines = []
    lines.append(
        f"Selected basis: {best['name']} (n_basis={best['n_basis']}, GCV={best['gcv']:.5f})."
    )

    if diag["lag1_autocorr"] > 0.3:
        lines.append(
            "Residuals show notable lag-1 autocorrelation, suggesting the fit "
            "may be slightly under-flexible for local structure in the data."
        )
    else:
        lines.append(
            "Residuals show low autocorrelation, indicating the fitted curve "
            "captures the local structure well without obvious under- or over-fitting."
        )

    if diag["oscillations"] > len(x) * 0.15:
        lines.append(
            "The fitted curve oscillates frequently, consistent with a basis "
            "well-suited to periodic or high-frequency structure."
        )
    elif diag["max_curvature_loc"] is not None:
        lines.append(
            f"The fit shows a localized region of high curvature near x="
            f"{diag['max_curvature_loc_display']:.2f}, consistent with a basis that "
            "adapts well to sharp local features."
        )

    if runner_up is not None and runner_up["gcv"] > 0:
        gap = runner_up["gcv"] - best["gcv"]
        rel_gap = gap / runner_up["gcv"] * 100
        lines.append(
            f"{best['name']} outperformed the next-best family ({runner_up['name']}) "
            f"by {rel_gap:.1f}% lower GCV."
        )

    return "\n".join(lines)


# ---------- Public entry point ----------

def analyze(x, y):
    """
    Main callable for the backend API.
    x, y: 1D lists or arrays of raw observations (any x range/scale).
    Returns a JSON-serializable dict.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Normalize x to [0, 1] so basis hyperparameters (knots, bandwidth, harmonics)
    # behave consistently regardless of the input's original scale
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else x - x_min

    result = select_best_basis(x_norm, y)
    rationale = generate_rationale(x_norm, y, result, x_original=x)

    leaderboard = [
        {"name": c["name"], "gcv": c["gcv"], "n_basis": c["n_basis"], "params": c["params"]}
        for c in sorted(result["best_per_family"].values(), key=lambda c: c["gcv"])
    ]

    return {
        "selected_basis": result["best"]["name"],
        "selected_params": result["best"]["params"],
        "gcv": result["best"]["gcv"],
        "rationale": rationale,
        "leaderboard": leaderboard,
        "fitted_curve": {
            "x": x.tolist(),  # original scale, for plotting
            "y_observed": y.tolist(),
            "y_fitted": result["best"]["y_hat"].tolist(),
        },
    }
