import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp

from sympy import symbols, sympify, lambdify, diff, latex, Rational

# ============================================================
# Page configuration and visual theme
# ============================================================

st.set_page_config(
    page_title="Numera Root Finder",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --numera-blue: #1557b0;
        --numera-navy: #12345b;
        --numera-light: #eef5ff;
        --numera-border: #bfd2ea;
        --numera-red: #c62828;
        --numera-red-bg: #fff3f3;
        --numera-green: #176b45;
        --numera-green-bg: #f1faf5;
        --numera-warning: #8a5a00;
        --numera-warning-bg: #fff8e8;
        }

    [data-testid="stAppViewContainer"] {
        background: #f7faff;
    }

    [data-testid="stHeader"] {
        background: #f7faff;
    }

    section[data-testid="stSidebar"] {
        background: #eef5ff;
        border-right: 1px solid var(--numera-border);
        width: 370px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 370px !important;
    }

    .numera-title {
        color: var(--numera-navy);
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.15rem 0;
    }

    .numera-subtitle {
        color: #52677f;
        font-size: 0.98rem;
        margin-bottom: 1.1rem;
    }

    .section-title {
        color: var(--numera-navy);
        font-weight: 700;
        font-size: 1.05rem;
        border-bottom: 2px solid #d4e2f4;
        padding-bottom: 0.35rem;
        margin-top: 0.8rem;
        margin-bottom: 0.65rem;
    }

    .notice {
        border-radius: 9px;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 0.75rem 0;
        border: 1px solid;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .notice.error {
        background: var(--numera-red-bg);
        border-color: #e7a9a9;
        color: #8f1d1d;
    }

    .notice.warning {
        background: var(--numera-warning-bg);
        border-color: #e6c777;
        color: #765000;
    }

    .notice.success {
        background: var(--numera-green-bg);
        border-color: #a8d7bc;
        color: var(--numera-green);
    }

    .notice.info {
        background: #f2f7fd;
        border-color: #bdd0e8;
        color: #274b70;
    }

    .cell-panel {
        background: white;
        border: 1px solid var(--numera-border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(18, 52, 91, 0.06);
        min-height: 180px;
    }

    .cell-panel-title {
        color: var(--numera-navy);
        font-size: 1.03rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .cell-panel-meta {
        color: #657a91;
        font-size: 0.86rem;
        margin-bottom: 0.8rem;
    }

    /* Keep calculator buttons readable in the sidebar. */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 43px;
        padding: 0.25rem 0.2rem;
        font-size: 0.86rem;
        font-weight: 600;
        white-space: nowrap;
        border-radius: 8px;
        border: 1px solid #b9cee6;
        background: #ffffff;
        color: #173b63;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--numera-blue);
        color: var(--numera-blue);
        background: #f7fbff;
    }

    /* Red/blue validation state for function input fields. */
    div.st-key-f_input input,
    div.st-key-g_input input {
        border: 2px solid var(--numera-border);
        border-radius: 9px;
    }

    div.st-key-f_input.numera-invalid input,
    div.st-key-g_input.numera-invalid input {
        border-color: var(--numera-red) !important;
        background: #fffafa !important;
        box-shadow: 0 0 0 1px rgba(198, 40, 40, 0.08);
    }

    div.st-key-f_input.numera-valid input,
    div.st-key-g_input.numera-valid input {
        border-color: #8fb4dc !important;
    }

    .root-card {
        margin: 18px 0 16px 0;
        padding: 18px 22px;
        border: 1px solid #b8d1eb;
        border-radius: 12px;
        background: linear-gradient(135deg, #f4f9ff, #e8f2fc);
        text-align: center;
    }
    .root-label {
        color: #315d8d;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .root-value {
        color: #123f6d;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .result-heading {
        color: var(--numera-navy);
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .stDataFrame {
        border: 1px solid #c9d9eb;
        border-radius: 10px;
        overflow: hidden;
        background: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="numera-title">Numera Root Finder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="numera-subtitle">Numerical methods for solving equations of the form f(x) = 0</div>',
    unsafe_allow_html=True,
)

x = symbols("x")

# ============================================================
# Supported functions
# ============================================================

FUNCTION_LOCALS = {
    "x": x,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda z: sp.log(z, 10),
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
}

# Button label, inserted expression, group.
KEYPAD_GROUPS = [
    (
        "Trigonometric",
        [
            ("sin", "sin("), ("cos", "cos("), ("tan", "tan("),
            ("cot", "cot("), ("sec", "sec("), ("csc", "csc("),
        ],
    ),
    (
        "Inverse Trigonometric",
        [
            ("sin⁻¹", "asin("), ("cos⁻¹", "acos("),
            ("tan⁻¹", "atan("),
        ],
    ),
    (
        "Exponential / Logarithmic",
        [
            ("exp", "exp("), ("ln", "log("),
            ("log₁₀", "log10("), ("√", "sqrt("),
            ("|x|", "Abs("), ("π", "pi"),
        ],
    ),
    (
        "Constants / Operators",
        [
            ("e", "E"), ("(", "("), (")", ")"),
            ("+", "+"), ("−", "-"), ("×", "*"),
            ("÷", "/"), ("x", "x"), ("xⁿ", "**"),
        ],
    ),
    (
        "Edit",
        [
            ("Clear", "__CLEAR__"), ("Backspace", "__BACKSPACE__"),
        ],
    ),
]

FUNCTION_NAMES = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "arcsin", "arccos", "arctan",
    "exp", "log", "ln", "sqrt", "abs", "Abs", "log10",
}


def append_token(expression, token):
    """Insert keypad tokens naturally, including implicit multiplication."""
    if token == "__CLEAR__":
        return ""
    if token == "__BACKSPACE__":
        return expression[:-1]

    existing = expression.rstrip()
    if not existing:
        return token

    previous = existing[-1]
    function_or_constant = (
        token.endswith("(")
        or token in {"x", "pi", "E"}
    )

    if function_or_constant and (
        previous.isdigit() or previous == ")" or previous in {"x", "E"}
    ):
        return existing + "*" + token

    if token == "(" and (previous.isdigit() or previous == ")" or previous in {"x", "E"}):
        return existing + "*("

    if token == "x" and (previous.isdigit() or previous == ")" or previous in {"E"}):
        return existing + "*x"

    return existing + token


# ============================================================
# Parsing, validation and numerical helpers
# ============================================================


def parse_expression(expr_str):
    return sympify(expr_str, locals=FUNCTION_LOCALS, evaluate=True)


def friendly_parse_error(expr_str, exc):
    text = str(exc)
    stripped = str(expr_str).strip()

    balance = stripped.count("(") - stripped.count(")")
    if balance > 0 or "unexpected eof" in text.lower() or "unexpected EOF" in text:
        return "ฟังก์ชันยังไม่สมบูรณ์ กรุณาตรวจสอบวงเล็บหรือเครื่องหมายที่ยังเปิดอยู่"

    if balance < 0:
        return "วงเล็บไม่สมดุล กรุณาตรวจสอบจำนวนวงเล็บเปิดและปิด"

    if "could not parse" in text.lower() or "syntax" in text.lower() or "token" in text.lower():
        return "รูปแบบฟังก์ชันไม่ถูกต้อง กรุณาตรวจสอบเครื่องหมาย ตัวเลข และชื่อฟังก์ชัน"

    return f"ไม่สามารถอ่านฟังก์ชันนี้ได้: {text}"


def validate_expression(expr_str, name):
    if not str(expr_str).strip():
        return None, False, f"กรุณาป้อน {name}"

    try:
        expr = parse_expression(expr_str)
    except Exception as exc:
        return None, False, friendly_parse_error(expr_str, exc)

    unknown_symbols = expr.free_symbols - {x}
    if unknown_symbols:
        names = ", ".join(sorted(str(s) for s in unknown_symbols))
        return (
            expr,
            False,
            f"{name} ใช้ตัวแปรที่ไม่รองรับ: {names}. ระบบรองรับเฉพาะตัวแปร x",
        )

    # Reject undefined user-created functions while allowing all functions
    # explicitly exposed through FUNCTION_LOCALS.
    undefined = [
        fn for fn in expr.atoms(sp.Function)
        if isinstance(fn.func, sp.core.function.UndefinedFunction)
    ]
    if undefined:
        return (
            expr,
            False,
            f"พบฟังก์ชันที่ระบบไม่รองรับใน {name}: {', '.join(sorted({str(fn.func) for fn in undefined}))}",
        )

    return expr, True, ""


def make_numeric_function(expr, vectorized=False):
    modules = ["numpy"] if vectorized else ["numpy", "math"]
    return lambdify(x, expr, modules=modules)


def safe_float(value):
    try:
        z = complex(value)
    except Exception as exc:
        raise ValueError(f"ผลลัพธ์ไม่สามารถตีความเป็นจำนวนจริงได้: {exc}") from exc

    if abs(z.imag) > 1e-10:
        raise ValueError("ผลลัพธ์เป็นจำนวนเชิงซ้อน ซึ่งโปรแกรมนี้รองรับจำนวนจริง")

    value = float(z.real)
    if not np.isfinite(value):
        raise ValueError("ผลลัพธ์ไม่เป็นจำนวนจำกัด")
    return value


def evaluate_function(func, value, function_name="f(x)"):
    try:
        result = func(value)
    except (ZeroDivisionError, FloatingPointError) as exc:
        raise ZeroDivisionError(
            f"{function_name} มีการหารด้วยศูนย์ที่ x = {value:g}"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"{function_name} ไม่อยู่ในโดเมนที่ x = {value:g}: {exc}"
        ) from exc
    except Exception as exc:
        text = str(exc).lower()
        if "divide" in text or "zero" in text:
            raise ZeroDivisionError(
                f"{function_name} มีการหารด้วยศูนย์ที่ x = {value:g}"
            ) from exc
        raise ValueError(
            f"ไม่สามารถคำนวณ {function_name} ที่ x = {value:g} ได้: {exc}"
        ) from exc

    try:
        return safe_float(result)
    except ZeroDivisionError:
        raise
    except ValueError as exc:
        raise ValueError(
            f"{function_name} ไม่สามารถให้ค่าจำนวนจริงที่ x = {value:g}: {exc}"
        ) from exc


def vectorized_values(func, xs, y_limit=1e6):
    with np.errstate(all="ignore"):
        try:
            values = np.asarray(func(xs), dtype=float)
        except Exception:
            values = np.full_like(xs, np.nan, dtype=float)

    values = np.where(np.isfinite(values), values, np.nan)
    values = np.where(np.abs(values) <= y_limit, values, np.nan)
    return values


def tolerance_from_dp(dp):
    return 0.5 * 10 ** (-dp)


def achieved_dp(error):
    if error is None or not np.isfinite(error) or error <= 0:
        return 10
    for d in range(10, 0, -1):
        if error < tolerance_from_dp(d):
            return d
    return 0


def same_value(a, b, tol=1e-14):
    try:
        return bool(np.isclose(float(a), float(b), rtol=0.0, atol=tol))
    except Exception:
        return False


def detect_divergence(history, stopping_tolerance):
    """Conservative numerical evidence for likely divergence/non-convergence."""
    if len(history) < 6:
        return False, ""

    vals = np.asarray(history, dtype=float)

    if not np.all(np.isfinite(vals)):
        return True, "ค่าประมาณไม่เป็นจำนวนจำกัด"

    if np.max(np.abs(vals)) > 1e10:
        return True, "ค่าประมาณมีขนาดเพิ่มขึ้นจนเกินขอบเขตเชิงตัวเลข"

    steps = np.abs(np.diff(vals))

    # Clearly expanding successive differences.
    if len(steps) >= 6:
        recent = steps[-6:]
        if np.all(recent[1:] > recent[:-1] * 1.05) and recent[-1] > 1e3:
            return True, "ความแตกต่างระหว่างค่าประมาณเพิ่มขึ้นอย่างต่อเนื่อง"

  
    # Persistent two-cycle oscillation.
    if len(vals) >= 8:
        recent = vals[-8:]
        even = recent[::2]
        odd = recent[1::2]

        even_scale = max(1e-12, 1e-4 * (1 + abs(np.mean(even))))
        odd_scale = max(1e-12, 1e-4 * (1 + abs(np.mean(odd))))
        cycle_gap = abs(np.mean(even) - np.mean(odd))

        if (
            np.std(even) < even_scale
            and np.std(odd) < odd_scale
            and cycle_gap > max(
                stopping_tolerance,
                even_scale,
                odd_scale,
            )
        ):
            return True, "ค่าประมาณเกิดการแกว่งเป็นคาบสองและไม่เข้าใกล้ค่าคงที่"

    return False, ""



def exceeds_iteration_bound(value, bound=1e10):
    """Return True when an iterate is too large for a meaningful numerical iteration."""
    try:
        z = float(value)
        return (not np.isfinite(z)) or abs(z) > bound
    except Exception:
        return True


def divergence_message(iteration, reason):
    return (
        f"การทำซ้ำลู่ออกในรอบที่ {iteration}: {reason} "
        "กรุณาพิจารณาเลือกค่าเริ่มต้นหรือรูปฟังก์ชัน g(x) ที่เหมาะสมกว่า"
    )


def generate_g_choices(f_expr):
    """Generate only fixed-point forms that can be verified as equivalent to f(x)=0."""
    if f_expr is None:
        return []

    candidates = [x - f_expr]

    try:
        poly = f_expr.as_poly(x)
        if poly is not None:
            deg = poly.degree()
            coeffs = poly.all_coeffs()

            if deg > 1:
                leading_term = coeffs[0] * x**deg
                rest = f_expr - leading_term
                candidates.append((-rest / coeffs[0]) ** Rational(1, deg))

            if deg >= 2 and len(coeffs) >= 2 and coeffs[-2] != 0:
                coeff_x1 = coeffs[-2]
                rest_x1 = f_expr - coeff_x1 * x
                candidates.append(-rest_x1 / coeff_x1)
    except Exception:
        pass

    candidates.extend([x + f_expr, x - sp.Rational(1, 2) * f_expr])

    result = []
    seen = set()
    for g in candidates:
        g_raw = str(g)
        if g_raw in seen or g_raw == "x":
            continue
        seen.add(g_raw)

        try:
            state, _ = check_fixed_point_form(f_expr, g)
        except Exception:
            state = False

        # Automatic options must be genuine functions of x and must not be
        # known-invalid fixed-point forms.
        if state is True and g.has(x) and g.is_real is not False:
            result.append({"raw": g_raw, "latex": latex(g)})

        if len(result) == 3:
            break

    return result


def _root_set_signature(solution_set):
    if isinstance(solution_set, sp.FiniteSet):
        values = []
        for item in solution_set:
            if item.is_real is False:
                continue
            try:
                values.append(float(sp.N(item)))
            except Exception:
                return None
        return tuple(sorted(round(v, 10) for v in values))
    return None


def check_fixed_point_form(f_expr, g_expr):
    """
    Return (state, message):
      True  -> symbolic evidence supports equivalence.
      False -> evidence shows the fixed-point equation is inconsistent.
      None  -> cannot prove equivalence symbolically.
    """
    h_expr = sp.together(x - g_expr)

    # A constant nonzero x-g(x) has no fixed point, while x-g(x)=0
    # makes every x a fixed point; neither is a valid root-finding form
    # for a nonzero equation f(x)=0.
    try:
        if not h_expr.has(x):
            if sp.simplify(h_expr) == 0:
                if sp.simplify(f_expr) == 0:
                    return True, ""
                return False, "g(x) = x ทำให้ทุกค่าของ x เป็น fixed point จึงไม่สมมูลกับ f(x) = 0"
            return False, "สมการ x = g(x) ไม่มีคำตอบ จึงไม่สอดคล้องกับการหารากของ f(x) = 0"
    except Exception:
        pass

    # Strong sufficient test: f(x) and x-g(x) differ by a nonzero constant.
    try:
        ratio = sp.simplify(sp.cancel(f_expr / h_expr))
        if ratio != 0 and not ratio.has(x) and ratio.is_finite is not False:
            return True, ""
    except Exception:
        pass

    # Polynomial/remainder test when both expressions are polynomial.
    try:
        if (
            f_expr.is_polynomial(x)
            and h_expr.is_polynomial(x)
            and sp.degree(h_expr, x) is not None
            and sp.degree(h_expr, x) >= 1
        ):
            remainder = sp.rem(f_expr, h_expr, x)
            if sp.simplify(remainder) == 0:
                return True, ""
    except Exception:
        pass

    # A useful exact test for common algebraic examples such as
    # x^3-x-2 = 0  <=>  x = cubert(x+2).
    try:
        if sp.count_ops(f_expr) <= 80 and sp.count_ops(h_expr) <= 80:
            f_roots = sp.solveset(f_expr, x, domain=sp.S.Reals)
            g_roots = sp.solveset(h_expr, x, domain=sp.S.Reals)
            fs = _root_set_signature(f_roots)
            gs = _root_set_signature(g_roots)
            if fs is not None and gs is not None:
                if fs == gs:
                    return True, ""
                return (
                    False,
                    "สมการ x = g(x) มีชุดคำตอบแตกต่างจาก f(x) = 0 "
                    "จึงไม่สามารถใช้รูป g(x) นี้แทนสมการเดิมได้",
                )
    except Exception:
        pass

    return (
        None,
        "ยังไม่สามารถยืนยันเชิงสัญลักษณ์ได้ว่า x = g(x) สมมูลกับ f(x) = 0 "
        "ระบบจะอนุญาตให้คำนวณ แต่ควรตรวจสอบรูป g(x) ทางคณิตศาสตร์เพิ่มเติม",
    )


# ============================================================
# Sidebar input and live validation
# ============================================================

def add_keypad_token(state_key, token):
    """Update a text-input widget's state through a Streamlit callback."""
    current = st.session_state.get(state_key, "")
    st.session_state[state_key] = append_token(current, token)

def clear_keypad_input(state_key):
    st.session_state[state_key] = ""

def backspace_keypad_input(state_key):
    current = st.session_state.get(state_key, "")
    st.session_state[state_key] = current[:-1]


if "f_input" not in st.session_state:
    st.session_state.f_input = "x**3 - x - 2"
if "g_input" not in st.session_state:
    st.session_state.g_input = "x - (x**3 - x - 2)"

with st.sidebar:
    st.markdown('<div class="section-title">Problem Input</div>', unsafe_allow_html=True)

    method = st.selectbox(
        "Method",
        ["Bisection", "Fixed-Point", "Newton-Raphson"],
    )

    # Function input
    f_str = st.text_input("Function f(x)", key="f_input")
    f_expr, f_valid, f_error = validate_expression(f_str, "f(x)")

    if not f_valid:
        st.markdown(
            f'<div class="notice error"><strong>Function input</strong><br>{f_error}</div>',
            unsafe_allow_html=True,
        )

    st.caption("Trigonometric functions use radians.")

    st.markdown("**Function keypad**")

    for group_index, (group_name, button_list) in enumerate(KEYPAD_GROUPS):
        st.caption(group_name)
        cols = st.columns(4)
        for button_index, (label, token) in enumerate(button_list):
            key = f"fkey_{group_index}_{button_index}"
            target = (
                clear_keypad_input if token == "__CLEAR__"
                else backspace_keypad_input if token == "__BACKSPACE__"
                else add_keypad_token
            )
            args = ("f_input",) if token in {"__CLEAR__", "__BACKSPACE__"} else ("f_input", token)
            cols[button_index % 4].button(
                label,
                key=key,
                on_click=target,
                args=args,
            )

    st.markdown('<div class="section-title">Numerical Settings</div>', unsafe_allow_html=True)

    accuracy_mode = st.radio(
        "Stopping criterion",
        ["Decimal Places (D.P.)", "Tolerance"],
        horizontal=True,
    )

    if accuracy_mode == "Decimal Places (D.P.)":
        decimals = st.slider(
            "Accuracy (D.P.)",
            min_value=1,
            max_value=10,
            value=6,
        )
        tolerance = tolerance_from_dp(decimals)
    else:
        tolerance = st.number_input(
            "Tolerance",
            min_value=1e-12,
            value=1e-6,
            step=1e-6,
            format="%.1e",
        )
        decimals = 6
        st.caption("Tolerance ต้องเป็นจำนวนบวก เพื่อใช้เป็นเกณฑ์ความคลาดเคลื่อนในการหยุดคำนวณ")

    max_iter = int(
        st.number_input(
            "Maximum iterations",
            min_value=1,
            max_value=100,
            value=100,
            step=1,
        )
    )

    input_errors = []
    input_warnings = []
    g_expr = None
    g_valid = True
    g_error = ""
    g_consistency = True
    g_consistency_message = ""

    if method == "Bisection":
        st.markdown('<div class="section-title">Initial Interval</div>', unsafe_allow_html=True)
        a_in = float(st.number_input("a (left endpoint)", value=1.0))
        b_in = float(st.number_input("b (right endpoint)", value=2.0))

        if a_in == b_in:
            input_errors.append("ต้องกำหนด a และ b ให้แตกต่างกัน")
        elif f_valid:
            try:
                f_func_check = make_numeric_function(f_expr)
                fa_check = evaluate_function(f_func_check, a_in, "f(x)")
                fb_check = evaluate_function(f_func_check, b_in, "f(x)")
                if fa_check != 0 and fb_check != 0 and fa_check * fb_check > 0:
                    input_errors.append(
                        "ไม่สามารถยืนยันการมีรากในช่วง [a,b] ได้ เนื่องจาก f(a) และ f(b) มีเครื่องหมายเดียวกัน "
                        "จึงไม่เป็นไปตามเงื่อนไขของวิธีแบ่งครึ่งช่วง"
                    )
            except (ZeroDivisionError, ValueError) as exc:
                input_errors.append(str(exc))

    elif method == "Fixed-Point":
        st.markdown('<div class="section-title">Fixed-Point Function</div>', unsafe_allow_html=True)
        try:
            parsed_f_for_g = f_expr if f_valid else None
            auto_choices = generate_g_choices(parsed_f_for_g) if parsed_f_for_g is not None else []
        except Exception:
            auto_choices = []

        options = [
            f"Option {i + 1}:  g(x) = {choice['latex']}"
            for i, choice in enumerate(auto_choices)
        ]
        custom_option = f"Option {len(auto_choices) + 1}:  Custom"
        options.append(custom_option)

        if not auto_choices and f_valid:
            st.info(
                "ไม่พบรูป g(x) ที่ระบบสามารถยืนยันได้ว่าเป็นรูปที่สมมูลกับ "
                "f(x) = 0 โดยอัตโนมัติ กรุณาเลือก Custom และกำหนด g(x) ด้วยตนเอง"
            )

        selected_option = st.radio("Fixed-point form", options)

        selected_auto_index = None
        for i in range(len(auto_choices)):
            if selected_option.startswith(f"Option {i + 1}:"):
                selected_auto_index = i
                break

        if selected_auto_index is not None:
            g_str = auto_choices[selected_auto_index]["raw"]
        else:
            g_str = st.text_input("Custom g(x)", key="g_input")

            g_expr, g_valid, g_error = validate_expression(g_str, "g(x)")
            if not g_valid:
                st.markdown(
                    f'<div class="notice error"><strong>Function input</strong><br>{g_error}</div>',
                    unsafe_allow_html=True,
                )

            st.caption("Custom g(x) keypad")
            for group_index, (group_name, button_list) in enumerate(KEYPAD_GROUPS):
                st.caption(group_name)
                cols = st.columns(4)
                for button_index, (label, token) in enumerate(button_list):
                    key = f"gkey_{group_index}_{button_index}"
                    target = (
                        clear_keypad_input if token == "__CLEAR__"
                        else backspace_keypad_input if token == "__BACKSPACE__"
                        else add_keypad_token
                    )
                    args = ("g_input",) if token in {"__CLEAR__", "__BACKSPACE__"} else ("g_input", token)
                    cols[button_index % 4].button(
                        label,
                        key=key,
                        on_click=target,
                        args=args,
                    )

        if selected_auto_index is not None:
            g_expr = parse_expression(g_str)
            g_valid = True
            g_error = ""


        x0 = float(st.number_input("Initial approximation x₀", value=1.0))

        if g_valid:
            try:
                g_func_check = make_numeric_function(g_expr)
                evaluate_function(g_func_check, x0, "g(x)")
            except (ZeroDivisionError, ValueError) as exc:
                input_errors.append(str(exc))

    else:
        st.markdown('<div class="section-title">Initial Approximation</div>', unsafe_allow_html=True)
        x0 = float(st.number_input("Initial approximation x₀", value=1.0))

        if f_valid:
            try:
                f_func_check = make_numeric_function(f_expr)
                df_expr_check = diff(f_expr, x)
                df_func_check = make_numeric_function(df_expr_check)
                evaluate_function(f_func_check, x0, "f(x)")
                dfx_check = evaluate_function(df_func_check, x0, "f'(x)")
                if abs(dfx_check) < np.finfo(float).eps:
                    input_errors.append(
                        "อนุพันธ์ f'(x₀) มีค่าเป็นศูนย์หรือใกล้ศูนย์มาก "
                        "จึงไม่สามารถใช้สูตรนิวตัน-ราฟสันจาก x₀ นี้ได้อย่างเหมาะสม"
                    )
            except (ZeroDivisionError, ValueError) as exc:
                input_errors.append(str(exc))

# Apply visible input border states via wrapper classes.
if not f_valid:
    st.markdown(
        "<style>div.st-key-f_input input { border-color: #c62828 !important; background:#fffafa !important; }</style>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<style>div.st-key-f_input input { border-color: #8fb4dc !important; }</style>",
        unsafe_allow_html=True,
    )

if method == "Fixed-Point" and "Custom" in selected_option:
    if not g_valid:
        st.markdown(
            "<style>div.st-key-g_input input { border-color: #c62828 !important; background:#fffafa !important; }</style>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<style>div.st-key-g_input input { border-color: #8fb4dc !important; }</style>",
            unsafe_allow_html=True,
        )

if input_errors:
    for item in input_errors:
        st.markdown(
            f'<div class="notice error"><strong>Input validation</strong><br>{item}</div>',
            unsafe_allow_html=True,
        )

if input_warnings:
    for item in input_warnings:
        st.markdown(
            f'<div class="notice warning"><strong>Mathematical verification</strong><br>{item}</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# Solver
# ============================================================


def run_solver():
    try:
        f_expr_local, valid, err = validate_expression(f_str, "f(x)")
        if not valid:
            return None, False, err, None, {}

        f_func = make_numeric_function(f_expr_local)
        data = []
        history = []
        found = False
        status = "not_converged"
        msg = ""
        detail = {
            "status": status,
            "history": history,
            "tolerance": tolerance,
            "f_expr": f_expr_local,
        }

        if method == "Bisection":
            a, b = sorted((float(a_in), float(b_in)))
            fa = evaluate_function(f_func, a, "f(x)")
            fb = evaluate_function(f_func, b, "f(x)")

            if fa == 0:
                data.append({
                    "Iteration": 1,
                    "a": a,
                    "b": b,
                    "f(a)": 0.0,
                    "f(b)": fb,
                    "x": a,
                    "f(x)": 0.0,
                    "Error": 0.0,
                })
                return pd.DataFrame(data), True, "", f_expr_local, {
                    **detail,
                    "status": "converged",
                    "reason": "พบรากที่ปลายช่วงด้านซ้าย",
                    "root": a,
                }

            if fb == 0:
                data.append({
                    "Iteration": 1,
                    "a": a,
                    "b": b,
                    "f(a)": fa,
                    "f(b)": 0.0,
                    "x": b,
                    "f(x)": 0.0,
                    "Error": 0.0,
                })
                return pd.DataFrame(data), True, "", f_expr_local, {
                    **detail,
                    "status": "converged",
                    "reason": "พบรากที่ปลายช่วงด้านขวา",
                    "root": b,
                }

            if fa * fb > 0:
                return None, False, (
                    "ไม่สามารถยืนยันการมีรากในช่วง [a,b] ได้ เนื่องจาก f(a) และ f(b) มีเครื่องหมายเดียวกัน"
                    "จึงไม่เป็นไปตามเงื่อนไขของวิธีแบ่งครึ่งช่วง"
                ), f_expr_local, {**detail, "status": "invalid_interval"}

            for i in range(max_iter):
                c = (a + b) / 2.0
                fc = evaluate_function(f_func, c, "f(x)")
                error = abs(b - a) / 2.0
                history.append(c)

                data.append({
                    "Iteration": i + 1,
                    "a": a,
                    "b": b,
                    "f(a)": fa,
                    "f(b)": fb,
                    "x": c,
                    "f(x)": fc,
                    "Error": error,
                })

                if abs(fc) < tolerance or error < tolerance:
                    found = True
                    status = "converged"
                    detail["root"] = c
                    break

                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc

            if not found:
                last_error = float(data[-1]["Error"])
                if accuracy_mode == "Decimal Places (D.P.)":
                    msg = (
                        f"ยังไม่ถึงเกณฑ์ {decimals} D.P. ภายใน {max_iter} รอบ; "
                        f"รอบที่ {max_iter} มีความถูกต้องประมาณ "
                        f"{achieved_dp(last_error)} D.P. "
                        f"(Error = {last_error:.{decimals}f})"
                    )
                else:
                    msg = (
                        f"ยังไม่ถึงเกณฑ์ Tolerance = {tolerance:.2e} ภายใน {max_iter} รอบ; "
                        f"รอบที่ {max_iter} มี Error = {last_error:.2e}"
                    )

        elif method == "Fixed-Point":
            g_expr_local, g_ok, g_err = validate_expression(g_str, "g(x)")
            if not g_ok:
                return None, False, g_err, f_expr_local, {**detail, "status": "invalid_g"}

            g_func = make_numeric_function(g_expr_local)
            curr_x = float(x0)

            for i in range(max_iter):
                iteration = i + 1
                try:
                    x_new = evaluate_function(g_func, curr_x, "g(x)")
                except ValueError as exc:
                    text = str(exc).lower()
                    if "too large" in text or "overflow" in text or "not a finite" in text or "ไม่เป็นจำนวนจำกัด" in text:
                        status = "diverged"
                        msg = divergence_message(
                            iteration,
                            "ค่าที่ได้จาก g(x) มีขนาดใหญ่เกินขอบเขตที่เหมาะสมสำหรับการคำนวณเชิงตัวเลข"
                        )
                        data.append({
                            "Iteration": iteration,
                            "x": curr_x,
                            "g(x)": np.nan,
                            "f(x)": np.nan,
                            "Error": np.nan,
                            "_x_old": curr_x,
                            "_x_new": np.nan,
                        })
                        break
                    raise

                # Check the new iterate before evaluating f(x_new).
                if exceeds_iteration_bound(x_new):
                    history.append(x_new if np.isfinite(x_new) else np.nan)
                    data.append({
                        "Iteration": iteration,
                        "x": curr_x,
                        "g(x)": x_new if np.isfinite(x_new) else np.nan,
                        "f(x)": np.nan,
                        "Error": abs(x_new - curr_x) if np.isfinite(x_new) else np.nan,
                        "_x_old": curr_x,
                        "_x_new": x_new if np.isfinite(x_new) else np.nan,
                    })
                    status = "diverged"
                    msg = divergence_message(
                        iteration,
                        "ค่าประมาณ xₙ มีขนาดเพิ่มขึ้นจนเกินขอบเขตที่เหมาะสมก่อนที่จะเข้าใกล้ราก"
                    )
                    break

                fx_new = evaluate_function(f_func, x_new, "f(x)")
                error = abs(x_new - curr_x)
                history.append(x_new)

                data.append({
                    "Iteration": iteration,
                    "x": curr_x,
                    "g(x)": x_new,
                    "f(x)": fx_new,
                    "Error": error,
                    "_x_old": curr_x,
                    "_x_new": x_new,
                })

                if abs(fx_new) < tolerance and error < tolerance:
                    found = True
                    status = "converged"
                    detail["root"] = x_new
                    break

                diverged, evidence = detect_divergence(history, tolerance)
                if diverged:
                    status = "diverged"
                    msg = divergence_message(iteration, evidence)
                    break

                curr_x = x_new

            if not found and status == "not_converged" and data:
                last_error = float(data[-1]["Error"])
                if accuracy_mode == "Decimal Places (D.P.)":
                    msg = (
                        f"ยังไม่ถึงเกณฑ์ {decimals} D.P. ภายใน {max_iter} รอบ; "
                        f"รอบที่ {max_iter} มีความถูกต้องประมาณ "
                        f"{achieved_dp(last_error)} D.P. "
                        f"(Error = {last_error:.{decimals}f})"
                    )
                else:
                    msg = (
                        f"ยังไม่ถึงเกณฑ์ Tolerance = {tolerance:.2e} ภายใน {max_iter} รอบ; "
                        f"รอบที่ {max_iter} มี Error = {last_error:.2e}"
                    )

            detail["g_expr"] = g_expr_local

        else:  # Newton-Raphson
            df_expr = diff(f_expr_local, x)
            df_func = make_numeric_function(df_expr)
            curr_x = float(x0)

            for i in range(max_iter):
                fx = evaluate_function(f_func, curr_x, "f(x)")
                dfx = evaluate_function(df_func, curr_x, "f'(x)")

                if abs(dfx) < np.finfo(float).eps:
                    status = "stopped"
                    msg = (
                        f"ไม่สามารถดำเนินการต่อในรอบที่ {i + 1}: "
                        "อนุพันธ์ f'(x) มีค่าเป็นศูนย์หรือใกล้ศูนย์มาก "
                        "จึงไม่สามารถหารในสูตรนิวตัน-ราฟสันได้อย่างเสถียร"
                    )
                    break

                try:
                    x_new = safe_float(curr_x - fx / dfx)
                except ValueError as exc:
                    text = str(exc).lower()
                    if "too large" in text or "overflow" in text or "not a finite" in text or "ไม่เป็นจำนวนจำกัด" in text:
                        data.append({
                            "Iteration": i + 1,
                            "x": np.nan,
                            "f(x)": np.nan,
                            "f'(x)": np.nan,
                            "Error": np.nan,
                            "_x_old": curr_x,
                            "_fx_old": fx,
                            "_dfx_old": dfx,
                        })
                        status = "diverged"
                        msg = divergence_message(
                            i + 1,
                            "ค่าประมาณใหม่มีขนาดใหญ่เกินขอบเขตที่สามารถใช้คำนวณเชิงตัวเลขได้"
                        )
                        break
                    raise

                # Check the new iterate before evaluating f(x_new) and f'(x_new).
                if exceeds_iteration_bound(x_new):
                    history.append(x_new if np.isfinite(x_new) else np.nan)
                    data.append({
                        "Iteration": i + 1,
                        "x": x_new if np.isfinite(x_new) else np.nan,
                        "f(x)": np.nan,
                        "f'(x)": np.nan,
                        "Error": abs(x_new - curr_x) if np.isfinite(x_new) else np.nan,
                        "_x_old": curr_x,
                        "_fx_old": fx,
                        "_dfx_old": dfx,
                    })
                    status = "diverged"
                    msg = divergence_message(
                        i + 1,
                        "ค่าประมาณ xₙ มีขนาดเพิ่มขึ้นจนเกินขอบเขตที่เหมาะสมก่อนที่จะเข้าใกล้ราก"
                    )
                    break

                fx_new = evaluate_function(f_func, x_new, "f(x)")
                dfx_new = evaluate_function(df_func, x_new, "f'(x)")
                error = abs(x_new - curr_x)
                history.append(x_new)

                data.append({
                    "Iteration": i + 1,
                    "x": x_new,
                    "f(x)": fx_new,
                    "f'(x)": dfx_new,
                    "Error": error,
                    "_x_old": curr_x,
                    "_fx_old": fx,
                    "_dfx_old": dfx,
                })

                if abs(fx_new) < tolerance and error < tolerance:
                    found = True
                    status = "converged"
                    detail["root"] = x_new
                    break

                diverged, evidence = detect_divergence(history, tolerance)
                if diverged:
                    status = "diverged"
                    msg = divergence_message(i + 1, evidence)
                    break

                curr_x = x_new

            if not found and status == "not_converged" and data:
                last_error = float(data[-1]["Error"])
                if accuracy_mode == "Decimal Places (D.P.)":
                    msg = (
                        f"ยังไม่ถึงเกณฑ์ {decimals} D.P. ภายใน {max_iter} รอบ; "
                        f"รอบที่ {max_iter} มีความถูกต้องประมาณ "
                        f"{achieved_dp(last_error)} D.P. "
                        f"(Error = {last_error:.{decimals}f})"
                    )
                else:
                    msg = (
                        f"ยังไม่ถึงเกณฑ์ Tolerance = {tolerance:.2e} ภายใน {max_iter} รอบ; "
                        f"รอบที่ {max_iter} มี Error = {last_error:.2e}"
                    )

            detail["df_expr"] = df_expr

        detail["status"] = status
        detail["history"] = history
        return pd.DataFrame(data), found, msg, f_expr_local, detail

    except ZeroDivisionError as exc:
        return None, False, str(exc), None, {"status": "domain_error"}
    except ValueError as exc:
        return None, False, str(exc), None, {"status": "domain_error"}
    except Exception as exc:
        return None, False, f"ไม่สามารถคำนวณได้: {exc}", None, {"status": "error"}


# ============================================================
# Cell calculation display
# ============================================================


def selected_cell_details(df, row_index, column, f_expr_local, method_local, g_expr_local=None):
    row = df.iloc[row_index]
    n = int(row["Iteration"])
    details = []
    d = decimals

    if column == "Iteration":
        return []

    if method_local == "Bisection":
        if column == "a":
            if n == 1:
                details.append(rf"a_1 = {row['a']:.{d}f}")
            else:
                prev = df.iloc[row_index - 1]
                if same_value(row["a"], prev["x"]):
                    details.append(rf"a_{{{n}}} = x_{{{n-1}}} = {row['a']:.{d}f}")
                elif same_value(row["a"], prev["a"]):
                    details.append(rf"a_{{{n}}} = a_{{{n-1}}} = {row['a']:.{d}f}")
                else:
                    details.append(rf"a_{{{n}}} = {row['a']:.{d}f}")

        elif column == "b":
            if n == 1:
                details.append(rf"b_1 = {row['b']:.{d}f}")
            else:
                prev = df.iloc[row_index - 1]
                if same_value(row["b"], prev["x"]):
                    details.append(rf"b_{{{n}}} = x_{{{n-1}}} = {row['b']:.{d}f}")
                elif same_value(row["b"], prev["b"]):
                    details.append(rf"b_{{{n}}} = b_{{{n-1}}} = {row['b']:.{d}f}")
                else:
                    details.append(rf"b_{{{n}}} = {row['b']:.{d}f}")

        elif column == "f(a)":
            details.extend([
                rf"f(a_{{{n}}}) = f({row['a']:.{d}f})",
                rf"= {row['f(a)']:.{d}f}",
            ])

        elif column == "f(b)":
            details.extend([
                rf"f(b_{{{n}}}) = f({row['b']:.{d}f})",
                rf"= {row['f(b)']:.{d}f}",
            ])

        elif column == "x":
            details.extend([
                rf"x_{{{n}}} = \frac{{a_{{{n}}}+b_{{{n}}}}}{{2}}",
                rf"= \frac{{{row['a']:.{d}f}+{row['b']:.{d}f}}}{{2}} = {row['x']:.{d}f}",
            ])

        elif column == "f(x)":
            details.extend([
                rf"f(x_{{{n}}}) = f({row['x']:.{d}f})",
                rf"= {row['f(x)']:.{d}f}",
            ])

        elif column == "Error":
            details.extend([
                rf"E_{{{n}}} = \frac{{b_{{{n}}}-a_{{{n}}}}}{{2}}",
                rf"= \frac{{{row['b']:.{d}f}-{row['a']:.{d}f}}}{{2}} = {row['Error']:.{d}f}",
            ])

    elif method_local == "Fixed-Point":
        x_old = float(row["_x_old"])
        x_new = float(row["_x_new"])

        if column == "x":
            if n == 1:
                details.append(rf"x_0 = {x_old:.{d}f}")
            else:
                details.append(rf"x_{{{n-1}}} = {x_old:.{d}f}")

        elif column == "g(x)":
            details.extend([
                rf"x_{{{n}}} = g(x_{{{n-1}}})",
                rf"= g({x_old:.{d}f}) = {x_new:.{d}f}",
            ])

        elif column == "f(x)":
            details.extend([
                rf"f(x_{{{n}}}) = f({x_new:.{d}f})",
                rf"= {row['f(x)']:.{d}f}",
            ])

        elif column == "Error":
            details.extend([
                rf"E_{{{n}}} = |x_{{{n}}}-x_{{{n-1}}}|",
                rf"= |{x_new:.{d}f}-{x_old:.{d}f}| = {row['Error']:.{d}f}",
            ])

    else:  # Newton-Raphson
        x_new = float(row["x"])
        x_old = float(row["_x_old"])
        fx_old = float(row["_fx_old"])
        dfx_old = float(row["_dfx_old"]) if "_dfx_old" in row.index else None

        if column == "x":
            details.append(rf"x_{{{n}}} = x_{{{n-1}}} - \frac{{f(x_{{{n-1}}})}}{{f'(x_{{{n-1}}})}}")
            if dfx_old is not None:
                details.append(
                    rf"= {x_old:.{d}f} - \frac{{{fx_old:.{d}f}}}{{{dfx_old:.{d}f}}} = {x_new:.{d}f}"
                )

        elif column == "f(x)":
            details.extend([
                rf"f(x_{{{n}}}) = f({x_new:.{d}f})",
                rf"= {row['f(x)']:.{d}f}",
            ])

        elif column == "f'(x)":
            fprime_value = float(row["f'(x)"])
            details.extend([
                rf"f'(x_{{{n}}}) = f'({x_new:.{d}f})",
                rf"= {fprime_value:.{d}f}",
            ])

        elif column == "Error":
            details.extend([
                rf"E_{{{n}}} = |x_{{{n}}}-x_{{{n-1}}}|",
                rf"= |{x_new:.{d}f}-{x_old:.{d}f}| = {row['Error']:.{d}f}",
            ])

    return details


# ============================================================
# Result table styling and graphing
# ============================================================


def table_format_map(df):
    formats = {}
    for col in df.columns:
        if col == "Iteration":
            formats[col] = "{:.0f}"
        else:
            formats[col] = f"{{:.{decimals}f}}"
    return formats


def graph_range(values, minimum_span=1.0):
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return -5.0, 5.0
    lo, hi = min(finite), max(finite)
    span = max(hi - lo, minimum_span)
    margin = max(0.15 * span, 0.5)
    lo -= margin
    hi += margin
    if lo == hi:
        lo -= 1
        hi += 1
    return lo, hi


def show_bisection_graph(df, f_expr_local):
    xvals = df["x"].astype(float).to_numpy()
    a_vals = df["a"].astype(float).to_numpy()
    b_vals = df["b"].astype(float).to_numpy()

    xmin, xmax = graph_range(np.concatenate([a_vals, b_vals]))
    xs = np.linspace(xmin, xmax, 700)
    f_vec = make_numeric_function(f_expr_local, vectorized=True)
    ys = vectorized_values(f_vec, xs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="#1557b0", width=2.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="#7d8ea3", line_width=1)
    f_scalar = make_numeric_function(f_expr_local)
    midpoint_y0 = evaluate_function(f_scalar, float(xvals[0]), "f(c)")
    fig.add_trace(
        go.Scatter(
            x=[xvals[0]], y=[midpoint_y0], mode="markers",
            marker=dict(size=10, color="#e45756"),
            name="Current midpoint (c, f(c))",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[a_vals[0], b_vals[0]], y=[0, 0], mode="lines+markers",
            line=dict(color="#6b91c5", width=5),
            marker=dict(size=8, color="#6b91c5"),
            name="Current interval [a, b]",
        )
    )

    frames = []
    for i in range(len(df)):
        frames.append(
            go.Frame(
                name=f"bisection_{i}",
                data=[
                    go.Scatter(
                        x=[xvals[i]],
                        y=[evaluate_function(f_scalar, float(xvals[i]), "f(c)")],
                        mode="markers",
                        marker=dict(size=10, color="#e45756"),
                        name="Current midpoint (c, f(c))",
                    ),
                    go.Scatter(
                        x=[a_vals[i], b_vals[i]],
                        y=[0, 0],
                        mode="lines+markers",
                        line=dict(color="#6b91c5", width=5),
                        marker=dict(size=8, color="#6b91c5"),
                        name="Current interval [a, b]",
                    ),
                ],
                traces=[1, 2],
            )
        )
    fig.frames = frames

    fig.update_layout(
        title="Bisection Iterations",
        xaxis_title="x",
        yaxis_title="f(x)",
        height=520,
        margin=dict(l=50, r=30, t=80, b=50),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.14,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Iteration: "},
                "steps": [
                    {
                        "label": str(i + 1),
                        "method": "animate",
                        "args": [[f"bisection_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for i in range(len(df))
                ],
            }
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


def show_fixed_point_graph(df, f_expr_local, g_expr_local):
    x_old = df["_x_old"].astype(float).to_numpy()
    x_new = df["_x_new"].astype(float).to_numpy()
    x0 = x_old[0]

    xmin, xmax = graph_range(np.concatenate([x_old, x_new]))
    xs = np.linspace(xmin, xmax, 700)
    g_vec = make_numeric_function(g_expr_local, vectorized=True)
    gs = vectorized_values(g_vec, xs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=gs, mode="lines", name="g(x)"))
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y = x"))

    path_x = [x0]
    path_y = [x0]
    cobweb_by_iteration = []
    prev = x0
    for xn in x_new:
        path_x.extend([prev, prev, xn])
        path_y.extend([prev, xn, xn])
        cobweb_by_iteration.append((list(path_x), list(path_y)))
        prev = xn

    fig.add_trace(
        go.Scatter(
            x=path_x[:1],
            y=path_y[:1],
            mode="lines+markers",
            name="Iterations",
        )
    )

    frames = []
    for i, (px, py) in enumerate(cobweb_by_iteration):
        frames.append(
            go.Frame(
                name=f"fixed_{i}",
                data=[go.Scatter(x=px, y=py, mode="lines+markers", name="Iterations")],
                traces=[2],
            )
        )
    fig.frames = frames

    fig.update_layout(
        title="Fixed-Point Iterations",
        xaxis_title="xₙ₋₁",
        yaxis_title="g(xₙ₋₁)",
        height=520,
        margin=dict(l=50, r=30, t=80, b=50),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.14,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Iteration: "},
                "steps": [
                    {
                        "label": str(i + 1),
                        "method": "animate",
                        "args": [[f"fixed_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for i in range(len(df))
                ],
            }
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


def show_newton_graph(df, f_expr_local):
    x_old = df["_x_old"].astype(float).to_numpy()
    x_new = df["x"].astype(float).to_numpy()

    f_func = make_numeric_function(f_expr_local)
    f_old = np.array([
        evaluate_function(f_func, float(v), "f(x)")
        for v in x_old
    ])

    xmin, xmax = graph_range(np.concatenate([x_old, x_new]))
    xs = np.linspace(xmin, xmax, 700)
    f_vec = make_numeric_function(f_expr_local, vectorized=True)
    ys = vectorized_values(f_vec, xs)

    df_expr_local = diff(f_expr_local, x)
    df_func = make_numeric_function(df_expr_local)

    # Initial traces: curve, current iterate, next approximation, tangent.
    xo0 = x_old[0]
    xn0 = x_new[0]
    fx0 = f_old[0]
    dfx0 = evaluate_function(df_func, xo0, "f'(x)")

    local_span = max(abs(xn0 - xo0) * 2.5, (xmax - xmin) * 0.18, 0.5)
    tangent_x0 = np.linspace(xo0 - local_span, xo0 + local_span, 160)
    tangent_y0 = fx0 + dfx0 * (tangent_x0 - xo0)
    tangent_y0 = np.where(
        np.isfinite(tangent_y0) & (np.abs(tangent_y0) <= 1e6),
        tangent_y0,
        np.nan,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines", name="f(x)",
            line=dict(color="#1557b0", width=2.5),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#7d8ea3", line_width=1)
    fig.add_trace(
        go.Scatter(
            x=[xo0], y=[fx0], mode="markers",
            marker=dict(size=10, color="#12345b", symbol="circle"),
            name="Current iterate (xₙ, f(xₙ))",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xn0], y=[0], mode="markers",
            marker=dict(size=10, color="#e45756", symbol="diamond"),
            name="Next approximation (xₙ₊₁, 0)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tangent_x0, y=tangent_y0,
            mode="lines",
            line=dict(color="#e45756", width=2.5),
            name="Tangent at current iterate",
        )
    )

    frames = []
    for i in range(len(df)):
        xo = x_old[i]
        xn = x_new[i]
        fx = f_old[i]
        dfx = evaluate_function(df_func, xo, "f'(x)")
        local_span = max(abs(xn - xo) * 2.5, (xmax - xmin) * 0.18, 0.5)
        tangent_x = np.linspace(xo - local_span, xo + local_span, 160)
        tangent_x = np.unique(np.append(tangent_x, xn))
        tangent_y = fx + dfx * (tangent_x - xo)
        tangent_y = np.where(
            np.isfinite(tangent_y) & (np.abs(tangent_y) <= 1e6),
            tangent_y,
            np.nan,
        )

        frames.append(
            go.Frame(
                name=f"newton_{i}",
                data=[
                    go.Scatter(
                        x=[xo], y=[fx], mode="markers",
                        marker=dict(size=10, color="#12345b", symbol="circle"),
                        name="Current iterate (xₙ, f(xₙ))",
                    ),
                    go.Scatter(
                        x=[xn], y=[0], mode="markers",
                        marker=dict(size=10, color="#e45756", symbol="diamond"),
                        name="Next approximation (xₙ₊₁, 0)",
                    ),
                    go.Scatter(
                        x=tangent_x, y=tangent_y,
                        mode="lines",
                        line=dict(color="#e45756", width=2.5),
                        name="Tangent at current iterate",
                    ),
                ],
                traces=[1, 2, 3],
            )
        )
    fig.frames = frames

    fig.update_layout(
        title="Newton-Raphson Iterations",
        xaxis_title="x",
        yaxis_title="f(x)",
        height=520,
        margin=dict(l=50, r=30, t=80, b=50),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.14,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Iteration: "},
                "steps": [
                    {
                        "label": str(i + 1),
                        "method": "animate",
                        "args": [[f"newton_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for i in range(len(df))
                ],
            }
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Calculate button and results
# ============================================================

input_signature = (
    f_str,
    method,
    accuracy_mode,
    decimals,
    float(tolerance),
    max_iter,
    round(a_in, 12) if method == "Bisection" else None,
    round(b_in, 12) if method == "Bisection" else None,
    g_str if method == "Fixed-Point" else None,
    round(x0, 12) if method in {"Fixed-Point", "Newton-Raphson"} else None,
)

can_calculate = f_valid and not input_errors and g_valid and g_consistency is not False

if st.button("Calculate", type="primary", disabled=not can_calculate, use_container_width=True):
    df_res, success, error_msg, f_expr_res, status_detail = run_solver()
    st.session_state.current_df = df_res
    st.session_state.success = success
    st.session_state.error_msg = error_msg
    st.session_state.f_expr = f_expr_res
    st.session_state.status_detail = status_detail
    st.session_state.result_signature = input_signature
    st.session_state.selected_cell = None

# Only show results that correspond to the current input state.
result_is_current = (
    "current_df" in st.session_state
    and st.session_state.get("result_signature") == input_signature
)

if result_is_current:
    df = st.session_state.get("current_df")
    success = st.session_state.get("success", False)
    error_msg = st.session_state.get("error_msg", "")
    status_detail = st.session_state.get("status_detail", {})
    f_expr_res = st.session_state.get("f_expr")

    status = status_detail.get("status", "")

    if success:
        st.markdown(
            f'<div class="notice success"><strong>สถานะการลู่เข้า</strong><br>'
            f'ค่าประมาณผ่านเกณฑ์การหยุดภายใน {len(df)} รอบ</div>',
            unsafe_allow_html=True,
        )
    elif error_msg:
        if status == "diverged":
            st.markdown(
                f'<div class="notice error"><strong>ตรวจพบพฤติกรรมการลู่ออก</strong><br>{error_msg}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="notice warning"><strong>สถานะการคำนวณ</strong><br>{error_msg}</div>',
                unsafe_allow_html=True,
            )

    if f_expr_res is not None:
        st.latex(rf"f(x) = {latex(f_expr_res)}")

    if accuracy_mode == "Decimal Places (D.P.)":
        criterion_text = f"เกณฑ์หยุด: {decimals} D.P. (Tolerance ภายใน = {tolerance:.2e})"
    else:
        criterion_text = f"เกณฑ์หยุด: Tolerance = {tolerance:.2e}"
    st.caption(criterion_text)

    # The root is the main output of this application. Show it prominently only
    # when the numerical method has actually met its stopping criterion.
    if success and status_detail.get("root") is not None:
        root_value = float(status_detail["root"])
        st.markdown(
            '<div class="root-card">'
            '<div class="root-label">Numerical Root</div>'
            f'<div class="root-value">{root_value:.{decimals}f}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.latex(rf"x^* \approx {root_value:.{decimals}f}")

    st.markdown('<div class="result-heading">ผลการคำนวณเชิงตัวเลข</div>', unsafe_allow_html=True)
    if df is not None and not df.empty:
        if method == "Bisection":
            display_columns = ["Iteration", "a", "b", "f(a)", "f(b)", "x", "f(x)", "Error"]
        elif method == "Fixed-Point":
            display_columns = ["Iteration", "x", "g(x)", "f(x)", "Error"]
        else:
            display_columns = ["Iteration", "x", "f(x)", "f'(x)", "Error"]

        display_df = df[display_columns]
        styled = display_df.style.format(table_format_map(display_df))

        event = st.dataframe(
            styled,
            use_container_width=True,
            height=min(620, 110 + 35 * len(df)),
            hide_index=True,
            on_select="rerun",
            selection_mode="single-cell",
            key="res_table",
        )

        selected_cells = list(event.selection.cells)
        if selected_cells:
            row_index, selected_column = selected_cells[0]
            st.session_state.selected_cell = (int(row_index), selected_column)

        st.markdown('<div class="section-title">Selected Cell</div>', unsafe_allow_html=True)
        detail_col, graph_col = st.columns([0.92, 1.65], gap="large")

        with detail_col:
            selected = st.session_state.get("selected_cell")
            if selected:
                row_index, selected_column = selected
                if selected_column == "Iteration":
                    st.caption("คอลัมน์ Iteration ใช้สำหรับระบุลำดับรอบการคำนวณ")
                elif 0 <= row_index < len(df) and selected_column in display_df.columns:
                    selected_row = display_df.iloc[row_index]
                    iteration = int(selected_row["Iteration"])
                    cell_value = selected_row[selected_column]

                    if method == "Fixed-Point":
                        g_expr_res = status_detail.get("g_expr", g_expr)
                    else:
                        g_expr_res = None

                    with st.container(border=True):
                        st.markdown(
                            '<div class="cell-panel-title">รายละเอียดการคำนวณ</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="cell-panel-meta">Column: <strong>{selected_column}</strong> &nbsp;|&nbsp; Iteration: <strong>{iteration}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        st.latex(str(cell_value) if selected_column == "Iteration" else f"{float(cell_value):.{decimals}f}")

                        formulas = selected_cell_details(
                            df,
                            row_index,
                            selected_column,
                            f_expr_res,
                            method,
                            g_expr_res,
                        )
                        for formula in formulas:
                            st.latex(formula)
            else:
                st.markdown(
                    '<div class="cell-panel">'
                    '<div class="cell-panel-title">รายละเอียดการคำนวณ</div>'
                    '<div class="cell-panel-meta">เลือกเซลล์ใดเซลล์หนึ่งในตารางเพื่อแสดงวิธีคำนวณของเซลล์นั้น</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        with graph_col:
            st.markdown('<div class="cell-panel-title">การแสดงขั้นตอนการวนซ้ำ</div>', unsafe_allow_html=True)
            st.caption("กราฟสามารถเล่นแบบภาพเคลื่อนไหวหรือเลือกดูทีละรอบได้")

            if method == "Bisection":
                show_bisection_graph(df, f_expr_res)
            elif method == "Fixed-Point":
                g_expr_res = status_detail.get("g_expr", g_expr)
                if g_expr_res is not None:
                    show_fixed_point_graph(df, f_expr_res, g_expr_res)
            else:
                show_newton_graph(df, f_expr_res)

st.markdown('<div class="section-title">ความรู้เพิ่มเติม</div>', unsafe_allow_html=True)
with st.expander("หลักการ ข้อดี ข้อจำกัด และการลู่เข้าของแต่ละวิธี", expanded=False):
    st.markdown(
        r"""
### วิธีแบ่งครึ่งช่วง (Bisection)

**หลักการ:** ใช้ช่วง $[a,b]$ ที่มี $f(a)f(b)<0$  แบ่งช่วงที่สนใจให้เล็กลงทีละครึ่ง 
แล้วตรวจสอบดูว่ารากอยู่บนช่วงย่อยใดของช่วงที่แบ่งครึ่งแล้ว จากนั้นแบ่งครึ่งของช่วงย่อยที่ได้ใหม่ 
ทำเช่นนี้ไปเรื่อยๆ ช่วงที่แบ่งจะมีขนาดเล็กลงเรื่อยๆ จนเข้าใกล้ศูนย์ ทำให้ค่าของ $f(x)$ มีค่าเข้าใกล้ศูนย์ด้วย 
ผลที่ได้คือ ค่ารากของฟังก์ชันที่ต้องการ

**ข้อดี**
- มีความมั่นคงสูงและความเสถียร
- ถ้า $f$ ต่อเนื่องบน $[a,b]$ และ$f(a)f(b)<0$ วิธีนี้จะรับประกันการเข้าใกล้ราก

**ข้อจำกัด**
- โดยทั่วไปลู่เข้าช้ากว่า Newton-Raphson และ Fixed-Point
- ต้องใช้ช่วง $[a,b]$ ที่มี $f(a)f(b)<0$ เท่านั้น

---

### วิธีจุดตรึง (Fixed-Point)

**หลักการ:** ถ้าฟังก์ชัน $g(x)$ มีจุดตรึงที่จุด $x = p$ แล้วฟังก์ชัน $f(x)$ ที่กำหนดโดย $f(x)=x - g(x)$
จะมีรากที่จุด $x = p$ เนื่องจาก $f(p)=0$

**ข้อดี**
- ลู่เข้าเร็วกว่าวิธี Bisection
- ไม่ต้องหาอนุพันธ์เหมือนวิธี Newton-Raphson

**ข้อจำกัด**
- การเลือก $g(x)$ มีผลต่อการลู่เข้า
- ค่าเริ่มต้นที่ไม่เหมาะสมอาจทำให้ลู่ออกหรือแกว่งโดยไม่เข้าใกล้ราก

**ทฤษฎีบทจุดตรึง (Fixed Point Theorem):** 

กำหนดให้ $g$ เป็นฟังก์ชันต่อเนื่องบนช่วง $[a,b]$ ซึ่ง $g(x) ∈ [a,b]$ สำหรับทุก $x ∈ [a,b]$
สมมติให้มีค่าคงที่ $k$ ซึ่ง $0 < k < 1$ ที่ทำให้

$$|g′(x)| \le k, x ∈ (a,b)$$

จะได้ว่า สำหรับจำนวน $p_0$ ใดๆ บนช่วง $[a,b]$ 
ลำดับของจุดที่กำหนดโดย 

$$p_n = g(p_{n-1}), n≥ 1$$ 

ลู่เข้าสู่ค่าจุดตรึง $p$ เพียงจุดเดียวบนช่วง $[a, b]$

---

### วิธีนิวตัน-ราฟสัน (Newton-Raphson)

**หลักการ:** ใช้เส้นสัมผัสของกราฟ $y=f(x)$ เพื่อสร้างค่าประมาณใหม่

$$x_{n}=x_{n-1}-\frac{f(x_{n-1})}{f'(x_{n-1})} , n \ge 1$$

**ข้อดี**
- ลู่เข้าเร็วมากเมื่อค่าเริ่มต้นอยู่ใกล้รากและเงื่อนไขเหมาะสม

**ข้อจำกัด**
- ต้องคำนวณอนุพันธ์ $f'(x)$
- ถ้า $f'(x_n)=0$ หรือมีค่าใกล้ศูนย์มาก การคำนวณอาจไม่เสถียร
- ค่าเริ่มต้นที่ไม่เหมาะสมอาจทำให้ลู่ออก

**เงื่อนไขของการลู่เข้า** 
จากทฤษฎีการลู่เข้าของจุดตรึงที่จะต้องมีค่า $|g'(x)| < 1$ เพื่อให้กราฟลู่เข้าหาคำตอบ จึงทำให้เกิดเงื่อนไข

$$\left|\frac{f(x)\cdot f''(x)}{[f'(x)]^2}\right| < 1$$

หรือ

$$|f(x)\cdot f''(x)| < [f'(x)]^2$$
        """
    )
