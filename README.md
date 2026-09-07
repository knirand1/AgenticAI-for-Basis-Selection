# Basis-Selection Agent

A small "agentic" data-analysis tool: given a set of noisy `(x, y)` observations, it automatically fits several different curve-fitting basis families, scores each one statistically, picks the best fit, and writes a plain-English explanation of why. It ships as a Python/FastAPI backend plus two independent frontends that call it.

## 1. Project overview

**Core purpose:** take a set of numeric observations and answer "what kind of function best describes this data, and why?" without the user having to know anything about splines, Fourier series, or model-selection statistics.

Concretely, the system:

1. Accepts a list of `x` values and corresponding `y` values (`backend/main.py`).
2. Fits that data using three different basis-function families — B-splines, a truncated Fourier series, and Gaussian radial basis functions (RBFs) — each tried at multiple hyperparameter settings (`backend/agent.py`).
3. Scores every fit with Generalized Cross-Validation (GCV), a way of estimating out-of-sample error without holding out a validation set.
4. Picks the lowest-GCV fit overall, and also the best fit *within each family* (for a "leaderboard" view).
5. Runs residual diagnostics on the winning fit (autocorrelation, curvature, oscillation count) and turns them into a short natural-language rationale for *why* that basis won.
6. Returns all of this — selected basis, its parameters, GCV score, rationale, per-family leaderboard, and the fitted curve — as JSON from a single `POST /analyze` endpoint.

There are two working frontends for this API:
- `docs/index.html` — a static, dependency-free HTML/JS page meant to be served from GitHub Pages. It is hard-coded to call a specific deployed backend (see "Two frontends" below).
- `frontend/` — a Next.js (App Router) + TypeScript + Tailwind CSS single-page app that calls a configurable backend URL, built for local development against your own running backend.

## 2. Algorithms explained

All of the algorithmic logic lives in **`backend/agent.py`**. `backend/main.py` is a thin FastAPI wrapper around it (request validation + error mapping) — no fitting or scoring logic lives there.

### 2.1 Basis expansion + least-squares fitting

The common pattern (`_fit_and_score`, `backend/agent.py` lines ~15-36) is: build a design matrix `X_design` whose columns are the basis functions evaluated at each `x`, solve for the coefficients that minimize squared error with `scipy.linalg.lstsq`, and compute the fitted values `y_hat = X_design @ coefs`. Three basis families are implemented, each producing its own design matrix:

- **B-spline basis** (`fit_bspline`, ~line 39): builds a cubic (degree-3 by default) B-spline basis over knots spaced evenly across the data's x-range, via `scipy.interpolate.BSpline`. Each basis function is constructed by setting a single coefficient to 1 and evaluating the resulting spline — i.e., the standard way to materialize a B-spline basis matrix. Tried with `n_knots` in `[5, 10, 15]`. B-splines are well-suited to data with **localized** structure (sharp bends in one region, smooth elsewhere), since each basis function only has local support.
- **Fourier basis** (`fit_fourier`, ~line 58): builds a design matrix of `sin(2πkx)` / `cos(2πkx)` terms for harmonics `k = 1..n_harmonics`, plus a constant term. Tried with `n_harmonics` in `[3, 6, 10]`. This basis is well-suited to **periodic** or globally oscillatory signals.
- **Gaussian RBF basis** (`fit_gaussian_rbf`, ~line 74): builds a design matrix of Gaussian bumps `exp(-(x - center)^2 / (2 * bandwidth^2))` centered at evenly spaced points, plus an intercept. Tried with `bandwidth` in `[0.05, 0.1, 0.2]` (10 centers, fixed). RBFs sit between the other two — smooth and flexible, without the strict locality of splines or the strict periodicity of Fourier terms.

Before fitting, `analyze()` (bottom of `agent.py`) normalizes `x` to the `[0, 1]` range. This is why the hyperparameter grids above (knot counts, bandwidths) are stated in normalized units — it keeps them meaningful regardless of the input data's actual scale.

**Guard rails:** each `fit_*` function checks that the number of basis functions it's about to build is strictly less than the number of observations, and raises `ValueError` if not (an underdetermined / rank-deficient fit produces an infinite or non-finite GCV score, which previously crashed JSON serialization — see the in-code comments and error messages in `agent.py`). `_fit_and_score` similarly raises if the degrees of freedom are exhausted or the resulting GCV isn't finite. `select_best_basis` catches and skips any candidate that raises, and raises its own `ValueError` if *every* candidate was skipped (i.e., not enough data for any basis family to be well-posed).

### 2.2 Model selection via Generalized Cross-Validation (GCV)

`_fit_and_score` (`backend/agent.py`, ~lines 15-36) scores each fit with GCV:

```
H = X_design @ pinv(X_design.T @ X_design) @ X_design.T   # the "hat" / smoother matrix
trace_H = trace(H)                                         # effective degrees of freedom
gcv = (n * RSS) / (n - trace_H) ** 2
```

GCV approximates leave-one-out cross-validation error in closed form, penalizing both poor fit (high residual sum of squares, `RSS`) and excessive model complexity (high effective degrees of freedom, `trace(H)`), without actually refitting the model `n` times. This is why it's a reasonable choice here: it lets `select_best_basis` (`agent.py`, ~line 100) compare fits from *entirely different* basis families — which don't have a common notion of "degrees of freedom" the way, say, polynomial order would — on a single, consistent scale, and pick whichever has the lowest score. The function selects both a single global winner (`best`) and the best configuration within each family (`best_per_family`), the latter feeding the API's `leaderboard` field.

### 2.3 Residual-diagnostics rationale generation

`_residual_diagnostics` and `generate_rationale` (`backend/agent.py`, ~lines 137-207) turn the winning fit's residuals into the human-readable `rationale` string returned by the API. It computes:

- **RMSE** — overall fit error.
- **Lag-1 autocorrelation** of the (mean-centered) residuals — a high value suggests the fit is missing structure the data actually has (under-fitting).
- **Max-curvature location** — where the fitted curve bends most sharply (via the second difference of the fitted values), mapped back to the original x-scale for display.
- **Oscillation count** — how many times the fitted curve's slope changes sign, i.e. how "wiggly" it is.

These diagnostics are then turned into a few sentences of prose: which basis was selected and its GCV; whether residual autocorrelation looks fine or suspicious; whether the fit is oscillatory (periodic-looking) or has one identifiable region of sharp curvature; and by how much (in relative GCV terms) the winner beat the runner-up family. This is templated text built from the computed diagnostics, not a call to any external language model.

## 3. Project structure

```
.
├── backend/              FastAPI service — all fitting/selection logic
│   ├── agent.py          Core algorithms (Section 2 above)
│   ├── main.py           FastAPI app: routes, request validation, error mapping
│   ├── requirements.txt  Python dependencies
│   └── runtime.txt       Python version pin (python-3.12.7) — a Render.com convention
├── docs/
│   └── index.html        Static, dependency-free frontend for GitHub Pages.
│                          Hard-coded to call a deployed backend
│                          (https://agenticai-for-basis-selection.onrender.com/analyze) —
│                          NOT the local backend by default.
├── frontend/              Next.js (App Router) + TypeScript + Tailwind CSS frontend
│   ├── app/               Route/page (page.tsx), root layout, global styles
│   ├── components/        CurveChart.tsx (inline SVG plot), Leaderboard.tsx (results table)
│   ├── lib/api.ts          Typed request/response contract + fetch call to the backend
│   └── .env.local.example  Template for the backend base URL (see Section 4)
└── README.md              This file
```

Two things worth calling out because they aren't obvious from the tree above:

- **Two independent frontends.** `docs/index.html` and `frontend/` both implement essentially the same one-page "submit x/y, see the result" flow against the same `/analyze` contract, but they are separate, unrelated codebases (the former is plain HTML/JS for GitHub Pages; the latter is the Next.js app). Changes to one do not affect the other.
- **Local, machine-generated directories are gitignored, not committed.** Python virtual environments (`backend/venv/`, or a `venv/` at the project root), `backend/__pycache__/`, `frontend/node_modules/`, `frontend/.next/`, and OS files like `.DS_Store` are all excluded by the root `.gitignore` (plus `frontend/.gitignore` for the Next.js-specific ones). This is why a fresh clone of this repo doesn't include them — see Section 4 for recreating them locally.

## 4. Getting started: install and run locally

This walks through everything needed to go from a fresh `git clone` to a working backend and frontend on your own machine. Nothing here is invented — every dependency, command, and file referenced below comes directly from `backend/requirements.txt`, `backend/runtime.txt`, `frontend/package.json`, and the source files themselves.

### 4.0 Prerequisites

- **Python 3** — `backend/runtime.txt` pins `python-3.12.7` (a Render.com deployment convention). Any reasonably recent Python 3 works locally; matching 3.12 avoids surprises.
- **Node.js + npm** — only needed if you want to run the `frontend/` Next.js app locally. `frontend/package.json` doesn't declare an `engines` field, so no specific Node version is enforced by the project itself; a current LTS release is a safe bet.
- **git**, to clone the repository.

There are no other services to install (no database, no message queue, etc.) — nothing in `backend/` or `frontend/` connects to one.

### 4.1 Clone the repository

```bash
git clone <this-repo-url>
cd AgenticAI-for-Basis-Selection
```

### 4.2 Set up and run the backend

The backend is required for real use of either frontend (the static `docs/index.html` page defaults to a deployed backend and can optionally be pointed at your local one — see 4.4).

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Leave this running in its own terminal. No environment variables or config files are required — there is no `.env` file, and nothing in `main.py` or `agent.py` reads from `os.environ`.

**Verify the backend is up** (in a second terminal):

```bash
curl http://localhost:8000/
# {"status": "ok", "message": "Basis-Selection Agent API is running."}
```

There is no automated test suite in this repository (no `tests/` directory, no `pytest`/`unittest` files) — the health check above, plus the `/analyze` call in 4.5, are the closest thing the codebase provides to "verification."

### 4.3 Set up and run the frontend (Next.js app)

In a new terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev
```

This opens the app at `http://localhost:3000`.

**Backend URL configuration:** the frontend reads `NEXT_PUBLIC_API_BASE_URL` (see `frontend/lib/api.ts`) to know which backend to call. It defaults to `http://localhost:8000` if unset — so if you started the backend as in 4.2, **no configuration step is required.** Only create an override if your backend runs somewhere else:

```bash
cp .env.local.example .env.local
# then edit .env.local, e.g.:
# NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### 4.4 (Optional) Use the static frontend instead

`docs/index.html` requires no install step — it's a single static file. Open it directly:

```bash
open docs/index.html      # macOS; on Linux: xdg-open docs/index.html
```

By default it calls the already-deployed backend at `https://agenticai-for-basis-selection.onrender.com/analyze` (the `API_URL` constant near the top of its `<script>` block) rather than your local server — so this option works with **zero local setup at all**, even without doing 4.2. To point it at your local backend instead, edit that constant to `http://localhost:8000/analyze` before opening the file (there's no env-var mechanism here, since it's plain static HTML).

### 4.5 End-to-end check

With the backend running (4.2) and either frontend open (4.3 or 4.4):

1. Submit the pre-filled sample data (Next.js app: click "Load sample data" then "Run agent"; static page: use its built-in example if present, or paste values).
2. Confirm the UI shows a selected basis family, a GCV score, a written rationale, a fitted-curve chart, and a leaderboard table — not an error state.

Or skip the UI and hit the API directly:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"x":[0,0.35,0.7,1.0,1.4,1.7,2.1,2.4,2.8,3.1,3.4,3.8,4.1,4.5,4.8,5.2,5.5,5.9,6.2,6.6],"y":[0.08,0.32,0.73,1.09,0.95,0.95,1.12,0.78,0.30,0.12,-0.37,-0.68,-0.80,-1.26,-1.25,-0.98,-0.85,-0.36,-0.21,0.05]}'
# -> a JSON object with selected_basis, gcv, rationale, leaderboard, fitted_curve
```

Note: `/analyze` requires at least 5 observations (`min_items=5` on the Pydantic request model), but the basis hyperparameter grids in `agent.py` need noticeably more than that to produce a well-posed fit — roughly 20+ points is a safe floor. With too few points, every candidate basis is underdetermined and the endpoint returns a `422` explaining that, instead of a result.

### 4.6 (Optional) Build for production

```bash
# backend: uvicorn's --reload is for development only; for production run e.g.
uvicorn main:app --host 0.0.0.0 --port 8000

# frontend
cd frontend
npm run build
npm run start
```

These commands come directly from `backend/main.py`'s own doc-comment and `frontend/package.json`'s `scripts` block, respectively; nothing beyond that (e.g. a specific process manager or hosting target) is specified in the code.

## 5. Ambiguities / things not fully specified in the code

- **No automated tests exist** for either the backend or the frontend. "Verification" above is manual (curl / browser) because that's the only mechanism the codebase itself provides.
- **No deployment config beyond `backend/runtime.txt`** is present — there's no `Procfile`, `Dockerfile`, or CI configuration in the repo, even though `docs/index.html` calls a Render-hosted URL. How that deployment is built/updated isn't evidenced in this codebase.
- **No `.env`/config file convention for the backend** — confirmed by grepping `main.py`/`agent.py` for environment-variable reads; there are none, so "configuration" for the backend is limited to the CLI flags passed to `uvicorn`.
