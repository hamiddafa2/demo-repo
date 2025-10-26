#!/usr/bin/env python3
"""
newton.py — Simple Newton–Raphson root finder with CLI args.

Examples:
  python newton.py --f "x**3 - x - 2" --x0 1.5
  python newton.py --f "cos(x) - x" --x0 1.0 --tol 1e-10
  python newton.py --f "x**3 - 2*x - 5" --x0 2 --df "3*x**2 - 2"
  python newton.py --f "exp(-x) - x" --x0 0.5 --max-iter 50 --alpha 1.0 --verbose
"""

import argparse, math, sys

def safe_eval(expr: str, x: float) -> float:
    """Evaluate expr at x using a restricted math environment."""
    if expr is None:
        raise ValueError("Expression is None.")
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["x"] = x
    return float(eval(expr, {"__builtins__": {}}, allowed))

def num_derivative(f_expr: str, x: float, h: float = 1e-6) -> float:
    """Centered finite-difference derivative."""
    return (safe_eval(f_expr, x + h) - safe_eval(f_expr, x - h)) / (2*h)

def newton(f_expr: str, x0: float, df_expr: str | None, tol: float,
           max_iter: int, alpha: float, verbose: bool):
    x = float(x0)
    for k in range(1, max_iter + 1):
        fx = safe_eval(f_expr, x)
        dfx = safe_eval(df_expr, x) if df_expr else num_derivative(f_expr, x)
        if dfx == 0 or abs(dfx) < 1e-14:
            return None, k, f"Derivative ~ 0 at iteration {k} (x={x:g})."

        step = alpha * fx / dfx
        x_new = x - step

        if verbose:
            print(f"iter {k:3d}: x={x: .12g}, f(x)={fx: .12g}, f'(x)={dfx: .12g}, step={step: .12g}")

        if abs(x_new - x) <= tol * max(1.0, abs(x_new)):
            return x_new, k, "converged"
        x = x_new

    return x, max_iter, "max_iter_reached"

def main():
    p = argparse.ArgumentParser(description="Newton–Raphson root finder with optional analytic derivative.")
    p.add_argument("--f", required=True, help='Function of x, e.g. "x**3 - x - 2" or "cos(x) - x"')
    p.add_argument("--df", default=None, help='Analytic derivative of f, e.g. "3*x**2 - 1". If omitted, uses numeric derivative.')
    p.add_argument("--x0", type=float, required=True, help="Initial guess.")
    p.add_argument("--tol", type=float, default=1e-8, help="Relative tolerance on x (default 1e-8).")
    p.add_argument("--max-iter", type=int, default=100, help="Max iterations (default 100).")
    p.add_argument("--alpha", type=float, default=1.0, help="Damping (0<alpha<=1), default 1.0.")
    p.add_argument("--verbose", action="store_true", help="Print iteration details.")
    args = p.parse_args()

    try:
        root, iters, status = newton(args.f, args.x0, args.df, args.tol, args.max_iter, args.alpha, args.verbose)
    except Exception as e:
        print(f"Error while evaluating expressions: {e}", file=sys.stderr)
        sys.exit(2)

    if status == "converged":
        print(f"\nRoot ≈ {root:.12g}  (iterations: {iters}, tol: {args.tol})")
        try:
            res = safe_eval(args.f, root)
            print(f"Check: f(root) ≈ {res:.3e}")
        except Exception:
            pass
        sys.exit(0)
    elif status == "max_iter_reached":
        print(f"\nStopped after {iters} iterations (max_iter reached). Best x ≈ {root:.12g}")
        try:
            res = safe_eval(args.f, root)
            print(f"f(x) ≈ {res:.3e}")
        except Exception:
            pass
        sys.exit(1)
    else:
        print(f"\nFailed: {status}")
        sys.exit(1)

if __name__ == "__main__":
    main()

