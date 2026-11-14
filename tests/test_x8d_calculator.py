"""Offline calculator, terminal CLI and tool-contract tests."""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from tools.x8d_calculator import calculate, fast_path_expression, tool_call


class CalculatorTest(unittest.TestCase):
    def test_exact_decimal(self):
        self.assertEqual(calculate("0.1 + 0.2")["result"], "0.3")
        self.assertFalse(calculate("0.1+0.2")["rounded"])

    def test_precedence_and_parentheses(self):
        self.assertEqual(calculate("2+3*4")["result"], "14")
        self.assertEqual(calculate("(2+3)*4")["result"], "20")
        self.assertEqual(calculate("-2^2")["result"], "-4")
        self.assertEqual(calculate("(-2)^2")["result"], "4")
        self.assertEqual(calculate("2^3^2")["result"], "512")

    def test_percentages(self):
        self.assertEqual(calculate("20% of 150")["result"], "30")
        self.assertEqual(calculate("200 + (200 * 10%)")["result"], "220")
        self.assertEqual(calculate("12.5% of 80")["result"], "10")

    def test_exact_fraction_and_rounding(self):
        result = calculate("1 / 3", precision=6)
        self.assertEqual(result["fraction"], "1/3")
        self.assertEqual(result["result"], "0.333333")
        self.assertTrue(result["rounded"])
        self.assertEqual(calculate("1/8", precision=8)["result"], "0.125")
        self.assertEqual(calculate("1/8", precision=1)["result"], "0.1")
        self.assertTrue(calculate("1/8", precision=1)["rounded"])

    def test_scientific_and_big_integer(self):
        self.assertEqual(calculate("1e3 + 2.5e-2")["result"], "1000.025")
        self.assertEqual(calculate("999999999999999999 * 2")["result"], "1999999999999999998")

    def test_errors_and_limits(self):
        for expr, code in [
            ("1/0", "division_by_zero"),
            ("2^0.5", "invalid_exponent"),
            ("2^101", "resource_limit"),
            ("1e999999", "resource_limit"),
            ("3 +", "invalid_expression"),
            ("__import__('os')", "invalid_expression"),
            ("2; rm -rf /", "invalid_expression"),
        ]:
            with self.subTest(expr=expr):
                self.assertEqual(tool_call({"expression": expr})["error"]["code"], code)
        self.assertEqual(tool_call({"expression": "1+1", "command": "ls"})["error"]["code"], "invalid_request")
        self.assertEqual(tool_call({"expression": "1+1", "precision": True})["error"]["code"], "invalid_precision")
        self.assertEqual(tool_call({"expression": "1" * 257})["error"]["code"], "resource_limit")

    def test_fast_path(self):
        self.assertEqual(fast_path_expression("What is 25 * 4?"), "25 * 4")
        self.assertEqual(fast_path_expression("calculate 20% of 150"), "20% of 150")
        self.assertEqual(fast_path_expression("0.1+0.2"), "0.1+0.2")
        self.assertIsNone(fast_path_expression("Plan the game after 2 moves"))
        self.assertIsNone(fast_path_expression("what is the weather?"))
        self.assertIsNone(fast_path_expression("2+ print(3)"))
        self.assertIsNone(fast_path_expression("hi"))

    def test_cli(self):
        p = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "x8d_calculator.py"), "0.1+0.2", "--json"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        self.assertEqual(json.loads(p.stdout)["result"], "0.3")
        p = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "x8d_calculator.py"), "1/0"],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(p.returncode, 2)
        self.assertIn("division_by_zero", p.stderr)


if __name__ == "__main__":
    unittest.main()
