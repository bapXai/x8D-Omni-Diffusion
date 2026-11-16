# coding=utf-8
"""Deterministic, bounded calculator for x8D's host tool boundary.

No Python eval, shell, model forward pass or binary floating point.
The exact internal number is a Fraction; decimal rendering is explicit.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import re
import sys
from typing import Any

MAX_EXPRESSION_CHARS = 256
MAX_LITERAL_DIGITS = 80
MAX_RESULT_BITS = 4096
MAX_EXPONENT_ABS = 100
MAX_SCI_EXP_ABS = 100
MAX_PRECISION = 60
DEFAULT_PRECISION = 24
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_PROMPT_RE = re.compile(r"^\s*(?:calculate|compute|what\s+is)\s+(.+?)\s*\??$", re.IGNORECASE)


class CalculationError(ValueError):
    """Safe error with machine-readable code for invalid arithmetic."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_RESULT_BITS:
        raise CalculationError("resource_limit", "arithmetic result exceeds bit-size limit")
    return value


def _lex(expression: str, *, enforce_limits: bool = True) -> list[str]:
    if not isinstance(expression, str) or not expression.strip():
        raise CalculationError("invalid_expression", "expression must be a non-empty string")
    if enforce_limits and len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculationError("resource_limit", "expression exceeds length limit")
    result: list[str] = []
    i = 0
    while i < len(expression):
        c = expression[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/()^%":
            if c == "*" and expression[i:i+2] == "**":
                result.append("^")
                i += 2
            else:
                result.append(c)
                i += 1
            continue
        if expression[i:i+2].lower() == "of" and (i+2 == len(expression) or not expression[i+2].isalnum()):
            result.append("of")
            i += 2
            continue
        match = _NUMBER_RE.match(expression, i)
        if match:
            literal = match.group(0)
            exponent_part = re.search(r"[eE]([+-]?\d+)$", literal)
            if enforce_limits and exponent_part and (len(exponent_part.group(1).lstrip("+-")) > 3 or abs(int(exponent_part.group(1))) > MAX_SCI_EXP_ABS):
                raise CalculationError("resource_limit", "scientific exponent exceeds limit")
            if enforce_limits and sum(ch.isdigit() for ch in literal) > MAX_LITERAL_DIGITS:
                raise CalculationError("resource_limit", "numeric literal exceeds digit limit")
            result.append(literal)
            i = match.end()
            continue
        raise CalculationError("invalid_expression", f"unsupported character at offset {i}")
    if enforce_limits and len(result) > 128:
        raise CalculationError("resource_limit", "too many expression terms")
    return result


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.at = 0

    def peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def pop(self) -> str:
        token = self.peek()
        if token is None:
            raise CalculationError("invalid_expression", "incomplete expression")
        self.at += 1
        return token

    def parse(self) -> Fraction:
        value = self.sum()
        if self.peek() is not None:
            raise CalculationError("invalid_expression", "unexpected trailing expression")
        return value

    def sum(self) -> Fraction:
        value = self.product()
        while self.peek() in ("+", "-"):
            op = self.pop()
            other = self.product()
            value = _bounded(value + other if op == "+" else value - other)
        return value

    def product(self) -> Fraction:
        value = self.unary()
        while self.peek() in ("*", "/", "of"):
            op = self.pop()
            other = self.unary()
            if op == "/" and other == 0:
                raise CalculationError("division_by_zero", "division by zero")
            value = _bounded(value / other if op == "/" else value * other)
        return value

    def unary(self) -> Fraction:
        if self.peek() == "+":
            self.pop()
            return self.unary()
        if self.peek() == "-":
            self.pop()
            return _bounded(-self.unary())
        return self.power()

    def power(self) -> Fraction:
        value = self.postfix()
        if self.peek() == "^":
            self.pop()
            exponent = self.unary()
            if exponent.denominator != 1:
                raise CalculationError("invalid_exponent", "exponent must be an integer")
            exp = exponent.numerator
            if abs(exp) > MAX_EXPONENT_ABS:
                raise CalculationError("resource_limit", "exponent exceeds limit")
            if value == 0 and exp < 0:
                raise CalculationError("division_by_zero", "zero cannot have a negative exponent")
            if max(value.numerator.bit_length(), value.denominator.bit_length()) * abs(exp) > MAX_RESULT_BITS:
                raise CalculationError("resource_limit", "exponentiation exceeds result limit")
            value = _bounded(value ** exp)
        return value

    def postfix(self) -> Fraction:
        value = self.atom()
        while self.peek() == "%":
            self.pop()
            value = _bounded(value / 100)
        return value

    def atom(self) -> Fraction:
        token = self.pop()
        if token == "(":
            value = self.sum()
            if self.pop() != ")":
                raise CalculationError("invalid_expression", "unclosed parenthesis")
            return value
        if not _NUMBER_RE.fullmatch(token):
            raise CalculationError("invalid_expression", "expected number or parenthesized expression")
        return _bounded(Fraction(token))


def _decimal(value: Fraction, precision: int) -> tuple[str, bool]:
    """Half-even rounding to precision decimal places; returns (display, exact)."""
    negative = value < 0
    scale = 10 ** precision
    scaled, remainder = divmod(abs(value.numerator) * scale, value.denominator)
    exact = remainder == 0
    if 2 * remainder > value.denominator or (2 * remainder == value.denominator and scaled % 2):
        scaled += 1
    digits = str(scaled)
    if precision:
        digits = digits.zfill(precision + 1)
        whole, tail = digits[:-precision], digits[-precision:].rstrip("0")
        result = whole + ("." + tail if tail else "")
    else:
        result = digits
    return ("-" if negative and scaled != 0 else "") + result, exact


def calculate(expression: str, precision: int = DEFAULT_PRECISION) -> dict[str, Any]:
    """Compute an exact rational and a bounded decimal representation."""
    if type(precision) is not int or not 0 <= precision <= MAX_PRECISION:
        raise CalculationError("invalid_precision", f"precision must be 0..{MAX_PRECISION}")
    number = _Parser(_lex(expression)).parse()
    decimal, exact = _decimal(number, precision)
    fraction = f"{number.numerator}/{number.denominator}" if number.denominator != 1 else str(number.numerator)
    return {"ok": True, "result": decimal, "fraction": fraction, "rounded": not exact, "precision": precision}


def tool_call(payload: Any) -> dict[str, Any]:
    """Typed host tool call: rejects extra keys, code and shell execution."""
    try:
        if not isinstance(payload, dict) or set(payload) - {"expression", "precision"}:
            raise CalculationError("invalid_request", "expected {expression, precision?} object")
        return calculate(payload.get("expression"), payload.get("precision", DEFAULT_PRECISION))
    except CalculationError as exc:
        return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}


def fast_path_expression(message: str) -> str | None:
    """Detect strictly arithmetic chat input; never guess from general prose."""
    if not isinstance(message, str):
        return None
    message = message.strip()
    match = _PROMPT_RE.fullmatch(message)
    candidate = match.group(1) if match else message
    if candidate.endswith("?"):
        candidate = candidate[:-1].strip()
    try:
        # Recognize the complete lexical input before applying evaluation limits.
        # This keeps excessive arithmetic on the tool path and rejects prose.
        tokens = _lex(candidate, enforce_limits=False)
    except CalculationError:
        return None
    if not tokens or (not match and not any(t in "+-*/^%" or t == "of" for t in tokens)):
        return None
    try:
        _Parser(_lex(candidate)).parse()
    except CalculationError as exc:
        # Mathematical errors still belong to the calculator, not the
        # synthetic model fallback. General prose/invalid syntax does not.
        return candidate if match or exc.code in ("division_by_zero", "invalid_exponent", "resource_limit") else None
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact CPU calculator; no model or CUDA required")
    parser.add_argument("expression", help="Arithmetic expression; use -- before leading negatives, e.g. -- '-2^2'")
    parser.add_argument("--json", action="store_true", help="Return a structured tool result")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal places (0..60)")
    args = parser.parse_args(argv)
    result = tool_call({"expression": args.expression, "precision": args.precision})
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    elif result["ok"]:
        print(result["result"])
    else:
        print(f"error ({result['error']['code']}): {result['error']['message']}", file=sys.stderr)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
