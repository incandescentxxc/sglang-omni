# Coding style

This document describes the coding style for human contributors and AI agents.
Sources: the [style prompt and do/don't examples](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5651721937)
and [additional anti-pattern examples](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5650093336),
including [discarded parameters](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5673011724).

### Principles

Write clean, professional, maintainable code. Match the surrounding codebase's conventions
where they exist; where they don't, follow these rules.

The overriding goal is simplicity: fewer, smaller files and fewer functions. Avoid
speculative generality.

### SIMPLICITY & NON-DUPLICATION

- Introduce a helper, wrapper, or abstraction layer only when it is used at least
  twice now and genuinely clarifies the call site.
- Don't reinvent what stdlib or an already-imported dep provides (`itertools`, `functools`,
  `collections`, `pathlib`, `dataclasses`, `pydantic`, `torch.nn.functional`). A 3-line
  wrapper around `lru_cache` is noise.
- Don't pre-extract `_helper`/`_impl` for "cleaner main flow" unless reused more than
  twice or too long to read in one screen. A 60-line top-to-bottom function beats
  three 20-line `_step_one/_two/_three` called once each.
- Don't add interfaces/base classes/registries/plugin systems before a second concrete
  implementation exists. Two cases first, then abstract.
- Reuse alone is not enough to justify a trivial helper: inline a two-line frame
  alignment or device/dtype accessor used only once or twice when it adds indirection.
  A nested callback scoped to a factory or graph capture can stay local.

### LANGUAGE & COMMENTS

- English only: comments, docstrings, log strings, CLI help. (User-facing translatable
  strings go through i18n — not this rule's concern.) If the repo is non-English, match it.
- Comments sparse, one-line, only for the genuinely non-obvious. Restating code is noise.
- NO process markers: no ★, `# P1`, `# [FIX]`, `# TODO` without a ticket, `# === SECTION ===`
  banners.
- NO provenance leakage: never name other repos/upstream/"the closed source" in source.
- NO verbose rationale in comments — the "why" lives in design docs / commits / PRs.
- Docstrings: Google-style, 1-3 lines. Args/Returns only when non-obvious. One short module
  docstring per file.
- Do not add opening comment blocks or lengthy explanations after class definitions.
  Use the short module docstring and brief explanations where needed.
- A long-term workaround gets a one-line comment naming the constraint.
- `# noqa: <code>` allowed with a reason; bare `# noqa` is not.

### NAMING

- Classes PascalCase; functions/variables snake_case; constants UPPER_SNAKE.
  Use ordinary names for module-level functions, classes, methods, and attributes;
  reserve a leading underscore for functions nested inside another function.
  Preserve language-defined special methods such as `__init__`.
- Names say what, not how: `load_checkpoint` not `do_thing`; `num_codebooks` not `n`.
  Single letters only for loop indices (`i`,`j`) or math (`x`,`y`,`t`).
- In model interfaces, names must identify the physical quantity, tensor role, or
  unit: `noisy_mel`, `token_condition`, `speaker_embedding`, `prompt_mel`,
  `mel_frame`, and `autocast_dtype`, rather than `x`, `mu`, `spks`, `cond`,
  `frames`, or `compute_dtype`. Short mathematical names belong in local equations.

### TYPING & SIGNATURES

- Full type hints, modern syntax: `X | Y`, `list[...]`, `dict[str, int]`, `X | None`.
  Annotate return types.
- ONE typing style per repo. Don't mix `Optional[X]`/`Union[X,Y]` with `X | Y`. Modern
  preferred; if the repo uses `Optional`, match it.
- Either `requires-python >= 3.10` (native `X | Y`) or `from __future__ import annotations`.
  Don't use `Optional` to work around forward refs — use quoted annotations
  (`"ModelConfig"`). Type-only imports are covered in IMPORTS.
- Closed value sets → `Literal[...]` or `Enum`, not bare strings in comparisons.
- No mutable function defaults: `def f(x=[])`/`= {}` are bugs. Use a `None` sentinel.
- Use concrete types, including model and decoder types, rather than `any`/`Any`
  or bare `dict`/`list`/`tuple`. Type resource handles precisely; for example, a
  graph-pool handle is `tuple[int, int] | None`.
- Do not accept a parameter only to immediately delete it to silence type or lint
  checks, such as starting a function with `del request_id`. Remove unnecessary
  parameters and update callers. If an established interface requires an unused
  parameter, preserve the contract and make that constraint explicit rather than
  pretending the parameter is used.

### DATA STRUCTURES

- Internal value objects (config, messages, state): `@dataclass`, prefer `kw_only=True`;
  mutable defaults via `field(default_factory=...)`.
- Cross-boundary schemas (API req/resp, untrusted input, needs validation): `pydantic.BaseModel`.
  Don't hand-roll validators pydantic gives free.
- Don't mix the two for one concept. Pick per role, not per mood.

### STRUCTURE

- File > ~400 lines → extract a module. But a 30-line file holding one `_helper` called once
  is also wrong — merge it. Related functions belong together; one concern per file
  does not mean one function per file.
- One entry-point mechanism per repo (hydra/argparse/fire/`[project.scripts]`). `if __name__
  == "__main__"` only in entry-point modules, never library modules.
- Module-level side effects (env mutation, global state, `setup_root`, resolver registration)
  only in entry-point modules. Library imports must be side-effect-free.
- No dead code: unreachable branches, commented-out blocks, stale TODOs,
  "kept for later" stubs.

### ERROR HANDLING & MATURITY

- `assert` for internal invariants (shapes, device, dtype, preconditions you control) — fail
  fast, loud, consistently across sibling modules.
- `raise ValueError/RuntimeError/FileNotFoundError` for user input / environment errors.
- No bare `except:`. No `except Exception as e: pass`. Catch specific exceptions; if broad,
  log with a reason or re-raise.
- Maturity progression — this matters:
  - Prototyping: try/except around the call you don't yet trust is fine.
  - Mature: remove it. Replace `if x is not None and isinstance(x, ...)` chains with an
    assert or a direct call. Defensive scaffolding (dummy forwards to align AllReduce,
    logger enable/disable gymnastics, type-coercion loops, broad try/except swallowing
    errors) is DEBT, not safety. Trust your invariants; let failures surface as stack traces.
  - Review: every try/except and defensive `if` must answer "what breaks if I delete this?"
    If the answer is "nothing, it was for debugging" — delete it.
- Do not wrap large blocks in `try/except` to guard against speculative, extremely
  unlikely failures. Handle errors that realistically occur on the main execution
  path; let unexpected failures surface.
- Do not turn model execution failures into fabricated outputs such as
  `torch.zeros_like(tokens)`. When the model contract guarantees a tensor, use
  `output = model(tokens)` directly; remove impossible `None` checks and redundant
  `isinstance`/`torch.tensor(output)` coercion. Validate genuinely untrusted results
  at their boundary instead of weakening an established internal contract.
- Access fields on known types directly. Do not use `getattr` defaults or `hasattr`
  probes to hide missing required attributes. Read `flow.pre_lookahead_len` directly;
  do not search multiple locations and substitute a default. Likewise, do not catch
  `AttributeError` around `next(flow.parameters())` when parameters are guaranteed.

### CONTROL FLOW

- Every `if` must have an `else`, either directly or through a complete
  `if/elif/else` chain.
- Validate inputs and preconditions before the main logic. Keep related checks
  flat and minimize nesting instead of interleaving checks with the work they guard.

### LOGGING

- ONE logging library per repo. Don't mix stdlib `logging` and `loguru`. Configure once.
- One logger per module: `logging.getLogger(__name__)` or `from loguru import logger`.
- Keep messages terse. No manual `[Info]`, `[Warn]`, or `[step N]` prefixes;
  use the logger's level and context fields.
- `print` only for CLI output the end user reads (`--help`, visualization, `__main__` demo).
  Runtime info — even debug — goes through the logger.
- Distributed: guard rank-zero via logger config or a `RankedLogger`-style adapter, not
  scattered `if is_main:`. (Training loops keep theirs; don't spread the pattern.)
- f-string vs `%-style`: pick one per repo. loguru → f-strings idiomatic; stdlib hot path →
  `logger.info("Loading: %s", path)`.

### CONFIG & MAGIC VALUES

- No hardcoded URLs, absolute paths, or hostnames. Required deployment values must
  come from validated configuration, without embedded machine-specific defaults.
- Replace unexplained magic numbers with named constants at module scope.
- Empirical constants named + provenance comment: `DECODE_TOKS_PER_SEC = 6.7  # measured on H20`.
- Use one config mechanism, following the repository's existing choice (Hydra /
  argparse+yaml / dataclass / env). If none exists, pick the lightest that fits.
  Do not stack config dataclasses,
  YAML loaders, validators, and CLI overrides when argparse + dict would do.
- Do not ship a YAML file that is "documentation only" and never loaded.
- Define a constant used by only one module in that module; do not create a
  cross-file import solely for it.

### IMPORTS

- Group: stdlib / third-party / local, blank-line separated, alphabetical within group.
- No `sys.path.insert` hacks. Use package imports, PYTHONPATH, or
  `pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)`
  in entry-point modules only. Use one setup mechanism per repo.
- No function-local imports except documented circular-dependency breaks. Put
  type-only cycle-breaking imports under `if TYPE_CHECKING:` with quoted annotations.
- Import from the defining module using the full package path, such as
  `from xxx.yy.zzz import kkk`, rather than through `__init__.py`. Keep package
  re-exports minimal and define an explicit `__all__`; no wildcard imports.

### TOOLING

- This repository already configures linting, formatting, and other checks in
  [.pre-commit-config.yaml](../.pre-commit-config.yaml). Run
  `pre-commit run --all-files` before completing a change.
- New logic gets a test. Test actual failure contracts and supported fallback paths;
  do not add tests solely to preserve speculative recovery scaffolding.
- Before completing a change, review it against the rules in this guide.

### TRAINING

Split training code into `train` and `trainer` scripts. Encapsulate training-related
functions in `trainer`; `train` should only call the trainer to run the training loop.
