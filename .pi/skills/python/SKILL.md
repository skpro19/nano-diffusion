---
name: python
description: >-
  Python style for nano-diffusion (keyword-only parameters, function signature
  layout, no dataclass, no Ruff). Use when writing, editing, or reviewing
  Python in this repository.
---

# Python style (nano-diffusion)

## General

- Do not use the `dataclass` decorator.
- Do not use Ruff for linting or formatting.

## Function and method signatures

- Use keyword-only arguments for functions/methods with **multiple** parameters: put a bare `*` after `self` (or after positional-only args), then name every remaining parameter so callers must pass them by keyword (e.g. `def append_step(self, *, obs: ..., action: ...) -> None`).
- Keep a function's return annotation on the `def` line when the whole signature fits one line (e.g. `def make_dagger_round_seeds(*, seed: int, rounds: int) -> list[int]:`).
- When a signature is broken across lines, put the closing `)`, `-> ReturnType`, and `:` on the **same line as the last parameter** — not on a separate line (e.g. `    epoch: int,) -> None:` with a trailing comma, or `    epoch: int) -> None:` without one); this keeps the full header collapsible as one block in the editor.

## Examples

One line:

```python
def episode_starts_from_ends(*, episode_ends: np.ndarray) -> np.ndarray:
```

Multi-line (closing on last parameter):

```python
def fit_normalization_statistics(
    *,
    state: zarr.Array,
    action: zarr.Array,
    episode_ends: np.ndarray,) -> NormalizationStatistics:
```

Avoid (closing paren on its own line):

```python
def example(
    *, a: int, b: int,
) -> None:
```
