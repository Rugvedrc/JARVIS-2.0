#!/usr/bin/env python3
"""Rugved Calculator CLI - A simple command-line calculator."""

import sys
import argparse


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def parse_expression(expr: str) -> tuple[float, str, float]:
    """Parse a simple infix expression like '2 + 3' or '5 * 4'.
    Returns (a, operator, b).
    Raises ValueError if the expression is malformed.
    """
    tokens = expr.strip().split()
    if len(tokens) != 3:
        raise ValueError("Expression must be in format: <num> <op> <num>")
    try:
        a = float(tokens[0])
        op = tokens[1]
        b = float(tokens[2])
    except ValueError:
        raise ValueError("Numbers must be valid")
    return a, op, b


def calculate(expr: str) -> float:
    """Calculate the result of a simple infix expression.
    Raises ValueError for unsupported operators or division by zero.
    """
    a, op, b = parse_expression(expr)
    if op == '+':
        return add(a, b)
    elif op == '-':
        return subtract(a, b)
    elif op == '*':
        return multiply(a, b)
    elif op == '/':
        return divide(a, b)
    else:
        raise ValueError(f"Unsupported operator: {op}")


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Rugved Calculator CLI - A simple command-line calculator."
    )
    parser.add_argument(
        "expression",
        nargs="?",
        help="Expression to evaluate, e.g., '2 + 3' or '5 * 4'"
    )
    args = parser.parse_args()

    if args.expression:
        try:
            result = calculate(args.expression)
            print(f"= {result}")
            sys.exit(0)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive mode
        print("Rugved Calculator CLI")
        print("Enter expressions like: 2 + 3 or 5 * 4")
        print("Type 'quit' or 'exit' to leave.\n")

        while True:
            try:
                expr = input("calc> ").strip()
                if expr.lower() in ('quit', 'exit'):
                    print("Goodbye!")
                    break
                if not expr:
                    continue
                result = calculate(expr)
                print(f"= {result}")
            except ValueError as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()