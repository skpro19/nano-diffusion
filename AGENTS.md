## Important Instructions
- **code readability** is paramount, do not compromise it for the sake of brevity or complexity

## General Instructions
- use Context7 when you are not sure or need official implementation details; if Context7 is unavailable, use the closest official upstream documentation source
- the git policy lives in `.cursor/skills/git/SKILL.md` — load it with `/skill:git` before committing or pushing
- for `.canvas.tsx` files, follow `.cursor/rules/canvas.mdc` (design template based on `cs231n-multi-head-attention.canvas.tsx`)

## Project Structure (planned)
- Goal: a minimal, readable Push-T Diffusion Policy based on `docs/diffusion.pdf`, using `datasets/pusht/pusht_cchi_v7_replay.zarr`.
- Use a small root-level Python package (not a `src/` layout):

```text
nano-diffusion/
├── pyproject.toml
├── nano_diffusion/
│   ├── __init__.py
│   ├── dataset.py     # Episode splits, sequence sampling, normalization
│   ├── model.py       # Conditional 1-D U-Net and, later, visual encoder
│   ├── policy.py      # Diffusion loss and action sampling
│   ├── train.py       # Training loop and checkpointing
│   └── evaluate.py    # Push-T rollouts and coverage metrics
├── configs/           # Training and evaluation configuration
├── tests/
│   └── test_dataset.py
├── scripts/           # Dataset verification, replay, and other utilities
├── datasets/          # Local data; ignored by Git
└── runs/              # Generated logs/checkpoints; ignored by Git
```

- This is the agreed target structure, not a claim that these modules already exist. Create modules only when needed; avoid empty scaffolding and premature subpackages.
- Start with `nano_diffusion/dataset.py` and `tests/test_dataset.py`: validate episode-level splits, training-only normalization statistics, and correctly aligned observation/action sequences before implementing the model.
- Build and validate a state-conditioned policy first, then add visual conditioning.
- Keep tests in `tests/`, standalone utilities in `scripts/`, and generated artifacts out of Git. Existing `checkpoints/` is also ignored.
- Keep human-facing setup and usage instructions in `README.md` as functionality becomes available.

## Pi Sub-agents
- When the `subagent` tool is available, delegate autonomously when it improves efficiency; explicit user requests are not required.
- Available user-level agents: `scout` for codebase exploration, `planner` for implementation plans, `worker` for implementation and tests, and `reviewer` for read-only code review.
- Run independent exploration tasks in parallel. Use a planner for complex changes and a reviewer after substantial code changes; handle simple tasks directly.
- Give each agent a self-contained task with relevant context, exact file paths, constraints, and expected output. Sub-agents do not inherit the conversation.
- Sub-agents share the working directory: avoid concurrent edits to the same files, preserve existing changes, and verify delegated work before reporting completion.
- If you are already executing a delegated sub-agent task, complete it directly rather than recursively delegating.

## GitHub README images
- GitHub can continue showing an older image when a README image is overwritten at the same URL. When changing a displayed PNG or GIF, give the updated image a new descriptive filename and update its README reference and generating script together; remove the obsolete image if it is no longer used.
- Before reporting an image update as live, verify the README reference and image blob on the pushed branch (which may not be the repository's default branch). Include the explicit branch URL when pointing someone to the result.

## AWS Administration
- source `.env` or always use `AWS_PROFILE=toy-pickplace-backup` when running `aws` CLI commands for S3; the profile covers `s3://toy-pickplace` and prefix-scoped access to `s3://toy-act` (`checkpoints/act_v1`, `checkpoints/act_v2`, `runs/act_v1`, `runs/act_v2`, `datasets`), region is `ap-south-1`
- `s3://toy-act` access is prefix-scoped, so a prefix that is not in the policy (for example a new act version) is denied even though other prefixes work; `AccessDenied` on list or `403` on read almost always means the prefix is missing from the policy, not bad credentials
- `aws configure export-credentials --profile <profile> --format env-no-export` emits `KEY=VALUE` lines suitable for `uv run --env-file`; never print, commit, or transfer the output outside the workload
- when an application profile lacks permission to update its own IAM policy, use `aws login --profile <admin-profile> --region <region>` only after the user explicitly authorizes browser-based authentication
- verify the authenticated principal with `aws sts get-caller-identity --profile <admin-profile>` before making changes
- apply the narrowest required IAM policy from a reviewed JSON file; never place credentials, account-specific tokens, or browser-login URLs in repository files
- verify the resulting S3 operations using the workload profile, not the administrative profile
- run `aws logout --profile <admin-profile>` after verification so administrative browser credentials do not remain active

### Updating an existing inline IAM policy
1. list available profiles: `aws configure list-profiles`
2. log in with admin: `aws login --profile <admin-profile> --region <region>`
3. verify identity: `aws sts get-caller-identity --profile <admin-profile>`
4. list the user's inline policies: `aws iam list-user-policies --profile <admin-profile> --user-name <user>`
5. inspect the current policy document: `aws iam get-user-policy --profile <admin-profile> --user-name <user> --policy-name <policy>`
6. update the policy inline (keep existing statements, add new ones): `aws iam put-user-policy --profile <admin-profile> --user-name <user> --policy-name <policy> --policy-document file://reviewed-policy.json`
7. wait a few seconds for IAM propagation (inline policies can take ~10s)
8. verify with the workload profile: e.g. `aws s3 ls --profile <workload-profile>`
9. log out admin: `aws logout --profile <admin-profile>`

Notes:
- the workload user has a 2048-byte total limit across all inline policies; if a new inline policy would exceed it (`LimitExceeded` on `put-user-policy`), update an existing inline policy or attach a version to a managed policy (`aws iam create-policy-version <policy-arn> --policy-document file://reviewed.json --set-as-default`)
- verify write and delete too, not just list: a small throwaway `put-object`/`delete-object` under the new prefix confirms the full grant before starting a long run


## uv commands
- use `uv` instead of `pip`
- use `uv add` to add dependencies
- use `uv run` for Python scripts, tests, and CLIs instead of calling `python` directly

## Vast.ai Administration
- API key is in `.env` as `VAST_API_KEY`
- The `vastai show user` CLI command is unreliable (returns 400); use the REST API instead
- Query balance via: `source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY" "https://console.vast.ai/api/v0/users/current"` — the `credit` field is the available balance

## Sections to ignore
Do not treat these as code changes; ignore them during development and review:
- `docs/` — manually curated documentation; not a code change
- `ablations/` — experiment results and run tracking; not production code
- `ISSUES.md` — personal issue tracker; not a code change
- `IDEAS.md` — personal idea log; not a code change
