# Prompt Optimizer

An iterative meta-prompt refinement system that uses Claude Opus 4.5 to automatically generate and optimize prompts based on evaluation feedback.

## Overview

Prompt Optimizer uses a two-phase optimization workflow to create high-quality zero-shot prompts:

1. **Training Phase:** Optimize prompt with full feedback until 100% training set pass rate
2. **Test Phase:** Validate on held-out test set with descriptive (non-specific) feedback to avoid overfitting

The system iteratively refines prompts until both training and test sets reach 100% pass rate, or until cost/iteration limits are reached.

## Key Features

- **Iterative Refinement:** Uses advanced LLMs for intelligent meta-prompt engineering
- **Model Flexibility:** Configurable optimizer and target models (Claude, GPT-4, Gemini, etc.)
- **Rich Feedback:** Comprehensive error analysis with diffs, categorization, and pattern detection
- **Overfitting Prevention:** Test feedback provides patterns without revealing specific examples
- **Cost Controls:** Hard limits on iterations, tokens, and plateau detection
- **Zero-Shot Output:** Generates standalone instruction prompts (no few-shot examples)

## Architecture

**Components:**
- **Optimizer:** Configurable LLM for meta-prompt engineering
- **Test Runner:** Executes prompts against evaluation cases
- **Feedback Analyzer:** Generates rich feedback (full for training, descriptive for test)
- **Orchestrator:** Controls optimization loop with stopping criteria

## Design Documents

Comprehensive design documentation is available in [`docs/plans/`](docs/plans/):

- **[prompt-optimization-system-design.md](docs/plans/prompt-optimization-system-design.md)** - Overall architecture and approach comparison
- **[implementation-plan.md](docs/plans/implementation-plan.md)** - Complete implementation plan with phases
- **[option-b-d-deep-analysis.md](docs/plans/option-b-d-deep-analysis.md)** - Technology stack analysis (DSPy vs LangSmith vs Custom)
- **[training-set-workflow.md](docs/plans/training-set-workflow.md)** - Data handling and workflow details
- **[test-feedback-strategy.md](docs/plans/test-feedback-strategy.md)** - Avoiding overfitting while reaching 100%
- **[iterative-refinement-implementation.md](docs/plans/iterative-refinement-implementation.md)** - Detailed specifications

## Project Status

🚧 **Currently in development** - Phase 1 (MVP) implementation starting

**Planned Features:**
- [x] Complete system design
- [x] Architecture documentation
- [ ] Phase 1: Core optimization loop (MVP)
- [ ] Phase 2: Rich feedback mechanisms
- [ ] Phase 3: Robustness (cost controls, checkpointing)
- [ ] Phase 4: Polish (CLI UI, examples, docs)
- [ ] Phase 5: LangSmith integration (future)

## Technology Stack

- **Language:** Python 3.10+
- **LLM SDK:** `anthropic` (official SDK)
- **CLI:** `typer`
- **Config:** `pydantic` + `pydantic-settings` + `yaml`
- **Diffing:** `deepdiff` + `difflib`
- **Output:** `rich` (beautiful CLI)
- **Testing:** `pytest`

## Estimated Metrics

- **Cost per optimization:** $10-20
- **Typical convergence:** 8-15 iterations
- **Time to completion:** 5-15 minutes
- **Code size:** ~2000-2500 LOC

## License

TBD

## Contributing

This project is currently in initial development. Design docs are complete and implementation is underway.
