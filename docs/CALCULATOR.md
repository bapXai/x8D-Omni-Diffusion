# Exact calculator host tool for x8D (#73)

## Why this tool exists

System 1 and System 2 should request **deterministic arithmetic**, not infer exact answers by token/byte generation. A terminal calculator performs arithmetic on the CPU with no model weights, CUDA, GPU, vLLM or new model-visible byte states. The host executes a validated tool request and returns its result to the model; see #67 for general tool invocation, observations and replay.

The implementation in tools/x8d_calculator.py uses Python's standard-library Fraction representation for exact rational operations. It does not call eval, execute Python snippets, use a shell, or import user-supplied code. For a trained model to learn when to call it, collect tool-use traces and validate tool-call schema under #67/#69.

## Terminal

Run from the repo root:

    python3 tools/x8d_calculator.py '0.1 + 0.2'
    # 0.3
    python3 tools/x8d_calculator.py '20% of 150'
    # 30
    python3 tools/x8d_calculator.py '1 / 3' --json
    # {"ok":true,"result":"0.333333333333333333333333","fraction":"1/3","rounded":true,"precision":24}

<!-- coderabbit-ai-generated:v1 -->

<!-- coderabbit-ai-generated-section:v1:b36aafd230c775323766251028e31ec5f0420adc5e4740505141565d9ca7a35c -->

For leading-negative expressions, put options before the `--` separator: `python3 tools/x8d_calculator.py --json -- '-2^2'` returns result `-4`.

Use argv arrays rather than shell interpolation in a tool runner.

<!-- /coderabbit-ai-generated:v1 -->

The CLI returns exit code 0 on success and 2 on invalid requests. Default output is only the result, not a reasoning trace. Structured JSON is available for tools and auditing.

Operators: +, -, *, /, ^ (or **), parentheses, scientific notation and postfix %. The word "of" is multiplication, e.g. "20% of 150". Percent is **always divided by 100**: to increase 200 by 10%, say "200 + (200 * 10%)" (result 220), not the ambiguous "200 + 10%".

Nonterminating decimals are displayed to 24 decimal places by default with half-even rounding; "fraction" is the exact answer and "rounded" discloses whether the decimal display lost precision. Use --precision N (0..60) to select decimal places. There is no implicit currency, unit conversion, symbolic algebra, trig, complex numbers, arbitrary code execution or unrestricted floating functions.

The parser caps input length, numeric literal sizes, exponent magnitude, internal result bits and output precision. Division by zero, invalid/unsupported expressions and resource exhaustion return explicit errors. No untrusted Python eval or command execution is possible.

## Typed model/host tool contract

    POST /v1/tools/calculate
    Content-Type: application/json

    {"expression":"20% of 150","precision":12}

Response:

    {"ok":true,"result":"30","fraction":"30","rounded":false,"precision":12}

Malformed requests return an HTTP 400 with a structured error, e.g. {"ok":false,"error":{"code":"division_by_zero","message":"division by zero"}}. The endpoint evaluates only a limited arithmetic grammar. Keep execution inside host policy/authorization boundaries; do not treat the existence of an endpoint as permission for arbitrary external actions.

The existing chat server has a deterministic **fast-path recognizer** for arithmetic-only messages and clear "calculate", "compute", and "what is" expressions. This bypasses the untrained synthetic text sampler. The fast path is a host rule—not proof that a trained System 1 expert selected the tool. Other language requests continue to the regular model path. A future trained x8D model can produce the same JSON argument contract and consume the returned bytes as a tool observation.

## Test

    python3 -m unittest tests.test_x8d_calculator tests.test_x8d_calculator_server -v
    python3 -m unittest discover -s tests -v

The first test covers arithmetic, precision, bounded resources, security and CLI. The second covers the actual HTTP handler and deterministic chat path. This calculator is deliberately independent of x8D compression, MoE activation, training and accelerator backends; compare a Rust port only after #70 supplies performance evidence.
