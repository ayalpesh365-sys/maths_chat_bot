from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from langchain_core.tools import tool
from rag_chain import search_theory_notes

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def parse_math_expression(expression: str):
    """
    Parse common student-style maths input into a SymPy expression.
    Supports examples like:
    - e^x
    - exp(I*theta)
    - cos(I*theta)
    - cos(i theta)
    - cos(iθ)
    """
    x = sp.symbols("x")
    theta = sp.symbols("theta", real=True)

    local_dict = {
        "x": x,
        "theta": theta,
        "I": sp.I,
        "i": sp.I,
        "e": sp.E,
        "E": sp.E,
        "pi": sp.pi,
        "exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
    }

    s = expression.strip()
    s = s.replace("θ", "theta")
    s = s.replace("Θ", "theta")
    s = s.replace("^", "**")

    # Common shorthand: i theta or I theta -> I*theta
    s = re.sub(r"\b[iI]\s+theta\b", "I*theta", s)

    return parse_expr(
        s,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


@tool
def theory_search_tool(query: str) -> str:
    """
    Search the PDF/text knowledge base for theory, formulas, assumptions,
    derivation steps, and applications.
    Use this before explaining or deriving mathematical concepts.
    """
    return search_theory_notes(query)


@tool
def taylor_series_tool(expression: str) -> str:
    """
    Compute a Maclaurin/Taylor series using SymPy.

    Input examples:
    exp(x)
    e^x
    sin(x)
    cos(x)
    exp(I*theta)
    cos(I*theta)
    cos(i theta)
    """

    x = sp.symbols("x")
    theta = sp.symbols("theta", real=True)

    try:
        expr = parse_math_expression(expression)

        if theta in expr.free_symbols:
            variable = theta
        elif x in expr.free_symbols:
            variable = x
        else:
            variable = sorted(expr.free_symbols, key=lambda s: s.name)[0]

        series = sp.series(expr, variable, 0, 8)

        return f"""
Expression:
{sp.pretty(expr)}

Expansion variable:
{variable}

Maclaurin series up to order 7:
{sp.pretty(series)}

LaTeX:
{sp.latex(series)}
"""

    except Exception as e:
        return f"Could not compute Taylor series. Error: {e}"


@tool
def symbolic_simplify_tool(expression: str) -> str:
    """
    Simplify or verify a symbolic expression using SymPy.

    Input examples:
    exp(I*theta) - cos(theta) - I*sin(theta)
    cos(I*theta) - cosh(theta)
    """

    try:
        expr = parse_math_expression(expression)
        simplified = sp.simplify(expr)

        return f"""
Original:
{sp.pretty(expr)}

Simplified:
{sp.pretty(simplified)}

LaTeX:
{sp.latex(simplified)}
"""

    except Exception as e:
        return f"Could not simplify expression. Error: {e}"


@tool
def plot_taylor_approximation_tool(function_name: str) -> str:
    """
    Plot a function and its Taylor approximations.
    Supported function_name values:
    sin, sin(x), sine, cos, cos(x), cosine, exp, exp(x), e^x
    """

    x = sp.symbols("x")
    name = function_name.strip().lower().replace(" ", "")
    name = name.replace("(x)", "")

    if name in {"sin", "sine"}:
        expr = sp.sin(x)
        display_name = "sin"
        x_values = np.linspace(-6, 6, 400)

    elif name in {"cos", "cosine"}:
        expr = sp.cos(x)
        display_name = "cos"
        x_values = np.linspace(-6, 6, 400)

    elif name in {"exp", "exponential", "e^x", "e**x"}:
        expr = sp.exp(x)
        display_name = "exp"
        x_values = np.linspace(-3, 3, 400)

    else:
        return "Supported functions are: sin, cos, exp."

    exact_func = sp.lambdify(x, expr, "numpy")

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, exact_func(x_values), label=f"{display_name}(x) exact")

    for order in [2, 4, 6]:
        approx = sp.series(expr, x, 0, order + 1).removeO()
        approx_func = sp.lambdify(x, approx, "numpy")
        plt.plot(x_values, approx_func(x_values), linestyle="--", label=f"Order {order}")

    plt.title(f"Taylor approximations for {display_name}(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()

    filename = PLOT_DIR / f"taylor_{display_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    return f"Plot saved at: {filename}"


@tool
def physics_application_tool(topic: str) -> str:
    """
    Connect a mathematical derivation topic to real physics applications.
    """

    topic_lower = topic.lower()

    if "taylor" in topic_lower:
        return """
Taylor series applications in physics:
1. Small-angle approximation in pendulum motion: sin(theta) ≈ theta.
2. Approximation near equilibrium points.
3. Perturbation theory in quantum mechanics.
4. Relativity approximations when v << c.
5. Optics and paraxial ray approximations.
"""

    if "euler" in topic_lower or "complex" in topic_lower:
        return """
Euler formula and complex exponential applications:
1. Oscillations and waves.
2. Fourier analysis.
3. Quantum wavefunctions.
4. Electromagnetic waves.
5. Signal processing.
"""

    return """
This mathematical topic can be connected to physics by identifying:
1. The mathematical model.
2. The physical assumptions.
3. The governing equation.
4. The physical system where it appears.
"""


@tool
def practice_question_tool(topic: str) -> str:
    """
    Generate derivation-focused practice and research-style questions.
    """

    return f"""
Practice questions for: {topic}

Beginner:
1. State the main formula or theorem used in this derivation.
2. Identify the assumptions behind the derivation.

Intermediate:
3. Derive the result step by step without skipping algebra.
4. Verify one step using differentiation, substitution, or simplification.

Advanced:
5. Extend the derivation to a complex variable or different expansion point.
6. Compare the exact function with its Taylor approximation.

Research-style:
7. Where does this approximation fail?
8. How is this derivation used in a physics model such as waves, oscillations, quantum mechanics, or relativity?
"""
