---
name: git
description: >-
  Git commit and push policy for this repo (conventional-commit subjects, merge
  commit format, signed commits, pre-work sync). Use before creating, amending,
  or pushing any commit, or when the user asks to commit or push.
---

# Git policy

## When to commit or push

- ONLY create commits when the user explicitly asks.
- ONLY push when the user explicitly asks.
- NEVER update git config.
- NEVER run destructive/irreversible git commands (force push, hard reset, etc.) unless the user explicitly requests them.
- NEVER skip hooks (`--no-verify`, etc.) unless the user explicitly requests it.
- NEVER force push to `main`/`master`; warn the user if they request it.
- If the commit message is wrong, fix it **before** pushing; never push then amend/force-push unless the user explicitly asks to rewrite published history.

## Commit messages

- Subject MUST be a single line: `<type>: <short summary>` (Conventional Commits).
- Common types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `ci`, `build`, `perf`, `revert`.
- Use plain types only — **no scopes** (`feat(scope): ...` is wrong).
- Summary after `type:` must start with a lowercase letter, have no trailing period, stay concise (about 6–12 words).
- Prefer intent/outcome over implementation detail; avoid vague subjects like `update`, `changes`, `misc`.
- Do **not** add a commit body unless the user explicitly asks for one.
- Before committing, check recent style: `git log --oneline -10` and match it.

## Merge commits

- Format: `merge: <branch-name>` (e.g. `merge: inference-fix`).
- Do not use default Git merge subjects like `Merge branch 'x' into dev`.

## Signing

- Use signed commits: `git commit -S`.
- At session start when committing, verify identity: `git config user.name`, `user.email`, `user.signingkey`.

## Pre-work sync

- Before significant code changes: `git fetch origin && git pull --ff-only`.

## Amend rules

- Avoid `git commit --amend` unless ALL apply: user requested amend (or pre-commit hook auto-modified files after a successful commit you made), HEAD was created by you in this conversation, commit has **not** been pushed.
- If commit failed or was rejected by a hook, fix the issue and create a **new** commit — do not amend.

## Examples

- `feat: add viewer camera capture and recalibrate top-camera`
- `fix: prevent calibration overwrite unless toggle was pressed`
- `docs: clarify host-client branch sync workflow`
