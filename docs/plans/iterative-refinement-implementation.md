# Iterative Meta-Prompt Refinement: Implementation Proposal

**Date:** 2026-01-25
**Status:** Design - Awaiting Technology Stack Decision
**Related:** [System Design Document](./2026-01-25-prompt-optimization-system-design.md)

## Overview

This document details the implementation proposal for **Approach 1: Iterative Meta-Prompt Refinement** - our recommended architecture for the prompt optimization system. This approach uses Claude Opus 4.5 as a meta-optimizer that iteratively refines prompts based on comprehensive evaluation feedback.

## Architecture Components

### 1. Optimizer Component
**Responsibility:** Generate and refine prompts using meta-reasoning

**Key Features:**
- Uses Claude Opus 4.5 for highest quality meta-prompt engineering
- Receives comprehensive feedback from eval runs
- Analyzes failure patterns and error categories
- Generates targeted improvements based on specific failure modes
- Maintains conversation history to track refinement trajectory

**Inputs:**
- Initial task description from user
- Eval results (scores, diffs, error categories)
- Previous prompt versions and their performance

**Outputs:**
- Refined prompt text
- Reasoning about changes made
- Confidence score (optional)

### 2. Test Runner Component
**Responsibility:** Execute prompts against evaluation suite

**Key Features:**
- Configurable target model (e.g., GPT-4, Claude Sonnet, Gemini)
- Runs prompt against all eval cases
- Handles both input/output pairs and programmatic assertions
- Captures actual outputs for comparison
- Timeout and error handling for misbehaving prompts

**Inputs:**
- Prompt to test
- Evaluation suite (user-defined)
- Target model configuration

**Outputs:**
- Pass/fail for each eval case
- Actual outputs from the prompt
- Execution metadata (latency, tokens used, errors)

### 3. Feedback Analyzer Component
**Responsibility:** Transform raw eval results into rich feedback signals

**Key Features:**
- **Score Calculation:** Compute pass rate (e.g., "7/10 passed")
- **Diff Generation:** Compare expected vs actual outputs
  - Character-level diffs for precise comparison
  - Semantic similarity scores for fuzzy matching
- **Error Categorization:** Classify failure types
  - Too verbose / too terse
  - Missing key information
  - Wrong format/structure
  - Incorrect reasoning/logic
  - Hallucination/factual errors
- **Trend Analysis:** Track improvement/regression across iterations

**Inputs:**
- Raw eval results from Test Runner
- Expected outputs from eval suite
- Historical performance data

**Outputs:**
- Structured feedback object containing:
  - Overall score and per-case results
  - Detailed diffs for failures
  - Categorized error analysis
  - Improvement suggestions

### 4. Orchestrator Component
**Responsibility:** Control the optimization loop and convergence

**Key Features:**
- Manages iteration loop
- Tracks convergence metrics
- Implements stopping criteria
- Handles state persistence across runs
- Provides progress reporting to user

**Stopping Criteria:**
- **Success:** All evals pass (100% success rate)
- **Max Iterations:** Configurable limit (default: 20)
- **Plateau Detection:** No improvement for N iterations (default: 5)
- **User Interruption:** Allow manual stop

**State Management:**
- Save each iteration (prompt, results, feedback)
- Enable resume from checkpoint
- Export optimization history for analysis

## System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                            │
│  - Task description                                          │
│  - Evaluation suite (input/output pairs + assertions)        │
│  - Target model configuration                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                             │
│  Iteration Loop (max 20 iterations)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │                               │
         │  Iteration N                  │
         │                               │
         │  ┌─────────────────────────┐  │
         │  │   1. OPTIMIZER          │  │
         │  │   (Claude Opus 4.5)     │  │
         │  │   Generate/Refine       │  │
         │  │   Prompt                │  │
         │  └──────────┬──────────────┘  │
         │             │                 │
         │             ▼                 │
         │  ┌─────────────────────────┐  │
         │  │   2. TEST RUNNER        │  │
         │  │   Execute prompt        │  │
         │  │   against evals         │  │
         │  └──────────┬──────────────┘  │
         │             │                 │
         │             ▼                 │
         │  ┌─────────────────────────┐  │
         │  │   3. FEEDBACK ANALYZER  │  │
         │  │   - Compute scores      │  │
         │  │   - Generate diffs      │  │
         │  │   - Categorize errors   │  │
         │  └──────────┬──────────────┘  │
         │             │                 │
         │             ▼                 │
         │       ┌──────────┐            │
         │       │ All Pass?│            │
         │       └─┬────┬───┘            │
         │         │Yes │No              │
         └─────────┼────┼────────────────┘
                   │    │
                   │    └─► Loop back to Optimizer
                   │         with feedback
                   ▼
         ┌─────────────────┐
         │  RETURN FINAL   │
         │  OPTIMIZED      │
         │  PROMPT         │
         └─────────────────┘
```

## Technology Stack Options

### Programming Language: Python ✓
**Selected:** Based on user preference and optimal fit for ML/AI domain

**Rationale:**
- Best ecosystem for AI/ML development
- Anthropic SDK officially supported and well-documented
- Rich libraries for evaluation, diffing, testing
- Easy prototyping and iteration
- Strong async support for API calls

### Framework Approach: Build from Scratch (Recommended)

**Option A: Build from Scratch** ⭐ **RECOMMENDED**

**Stack:**
- **LLM SDK:** `anthropic` (official Python SDK)
- **CLI Framework:** `click` or `typer` (modern, type-safe CLI)
- **Configuration:** `pydantic` (validation) + `yaml` or `toml` (config files)
- **Evaluation:** Custom eval runner
- **Diffing:** `difflib` (stdlib) + `deepdiff` (structured diffs)
- **Output Formatting:** `rich` (beautiful CLI output)
- **Async:** `asyncio` (concurrent eval runs)
- **Testing:** `pytest` + `pytest-asyncio`

**Pros:**
- Full control over optimization logic
- Simple, transparent architecture (~500-1000 LOC)
- No framework lock-in or version conflicts
- Easy debugging and customization
- Custom feedback mechanisms built exactly to spec
- Clean separation for future API layer

**Cons:**
- Need to implement eval harness from scratch
- No pre-built optimizers or samplers
- Manual prompt template management

**Implementation Complexity:** Low-Medium

---

**Option B: DSPy Framework**

**Stack:**
- **Framework:** `dspy-ai`
- **LLM SDK:** Integrated multi-provider support
- **Optimizers:** Built-in `BootstrapFewShot`, `MIPRO`, etc.

**Pros:**
- Mature framework from Stanford
- Built-in optimizers and compilers
- Good evaluation primitives
- Active community and examples

**Cons:**
- Designed for few-shot example optimization, not iterative prompt text refinement
- Our use case (meta-prompt refinement) doesn't fit DSPy's module composition model
- Would need to work around framework abstractions
- Adds complexity for features we don't need
- Comprehensive feedback (diffs, error categorization) would be custom anyway

**Implementation Complexity:** Medium (fighting abstractions)

---

**Option C: TextGrad Framework**

**Stack:**
- **Framework:** `textgrad`
- **Concept:** Differentiable programming for text
- **Optimizers:** Gradient-based text optimization

**Pros:**
- Novel approach treating LLMs like neural nets
- Automatic differentiation for text
- Research-backed methodology

**Cons:**
- Research-level code, less production-ready
- "Differentiable text" abstraction adds complexity we don't need
- Our comprehensive feedback is more direct than TextGrad's gradients
- Smaller community, less documentation
- Overkill for our use case

**Implementation Complexity:** High (steep learning curve)

---

**Option D: LangChain + LangSmith**

**Stack:**
- **Framework:** `langchain` + `langsmith`
- **Evaluation:** LangSmith eval tools
- **LLM SDK:** Multi-provider through LangChain

**Pros:**
- Comprehensive ecosystem
- Built-in evaluation and tracing
- Good observability with LangSmith

**Cons:**
- Heavy abstraction layers (chains, agents, etc.) not needed
- LangSmith requires paid service for serious usage
- Optimization logic would still be custom
- Framework overhead for simple use case
- Harder to reason about what's happening under the hood

**Implementation Complexity:** Medium-High (navigating abstractions)

---

## Recommended Technology Stack (Option A)

### Core Dependencies
```python
# LLM Interaction
anthropic==0.40.0           # Official Anthropic SDK

# CLI & Configuration
typer==0.12.0               # Modern CLI framework
pydantic==2.8.0             # Configuration validation
pydantic-settings==2.4.0    # Settings management
pyyaml==6.0.1               # YAML config files

# Output & Formatting
rich==13.7.0                # Beautiful terminal output
rich-click==1.8.0           # Rich formatting for Click/Typer

# Evaluation & Comparison
deepdiff==7.0.1             # Structured diff for complex objects
difflib (stdlib)            # Basic text diffing
jinja2==3.1.4               # Template rendering for prompts

# Async & Concurrency
asyncio (stdlib)            # Async API calls
aiofiles==24.1.0            # Async file I/O

# Testing & Development
pytest==8.3.0
pytest-asyncio==0.23.0
pytest-cov==5.0.0
```

### Project Structure
```
prompt-optimizer/
├── src/
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── meta_optimizer.py      # Opus-based prompt refiner
│   │   ├── prompts.py              # Meta-prompt templates
│   │   └── models.py               # Pydantic models
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── test_runner.py          # Eval execution
│   │   ├── feedback_analyzer.py    # Diff + error categorization
│   │   └── eval_types.py           # Input/output + assertion types
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── optimization_loop.py    # Main loop controller
│   │   ├── state_manager.py        # Persistence & checkpointing
│   │   └── stopping_criteria.py    # Convergence detection
│   └── cli/
│       ├── __init__.py
│       ├── main.py                 # CLI entry point
│       └── commands.py             # CLI commands
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── examples/
│   ├── sentiment_analysis/
│   │   ├── evals.yaml
│   │   └── config.yaml
│   └── code_generation/
│       ├── evals.yaml
│       └── config.yaml
├── docs/
│   └── plans/                      # Design docs (this file)
├── pyproject.toml                  # Modern Python packaging
├── README.md
└── .env.example                    # API key template
```

### Configuration File Format (YAML)
```yaml
# config.yaml
task_description: |
  Generate sentiment analysis for product reviews.
  Output should be: positive, negative, or neutral.

target_model:
  provider: anthropic
  model: claude-sonnet-4.5
  temperature: 0.7
  max_tokens: 500

optimizer:
  model: claude-opus-4.5
  max_iterations: 20
  plateau_threshold: 5

evaluations:
  - type: input_output
    cases:
      - input: "This product is amazing!"
        expected: "positive"
      - input: "Terrible quality, broke immediately"
        expected: "negative"
      - input: "It's okay, nothing special"
        expected: "neutral"

  - type: assertion
    description: "Output must be one of: positive, negative, neutral"
    code: |
      assert output.strip().lower() in ['positive', 'negative', 'neutral']
```

## Key Design Decisions

### 1. Why Opus 4.5 for Meta-Optimization?
- Highest quality reasoning for prompt engineering
- Excellent at analyzing failure patterns
- Strong instruction following for systematic refinement
- Cost justified by sample efficiency (5-10 iterations vs 50-100)

### 2. Why Configurable Target Model?
- Users may want to optimize for different deployment targets
- Allows testing against actual production model
- Can use cheaper models for eval runs (cost optimization)
- Enables comparison across models

### 3. Why Separate Feedback Analyzer?
- Modular design - can improve categorization independently
- Reusable across different optimization strategies
- Clear separation of concerns
- Easy to add new error categories or diff methods

### 4. Why YAML for Configuration?
- Human-readable for eval cases
- Easy to version control
- Natural for hierarchical configuration
- Can include multi-line strings (prompts, assertions)

### 5. CLI-First, API-Ready Design
- Start simple with CLI for immediate usability
- Design with clean interfaces for future API layer
- State management enables both sync (CLI) and async (API) use
- Components are already modular and testable

## Development Phases

### Phase 1: Core Loop (MVP)
- Implement basic optimization loop
- Simple optimizer using Opus
- Basic test runner for input/output pairs
- Console output for results
- **Goal:** End-to-end working system

### Phase 2: Rich Feedback
- Implement feedback analyzer
- Add diff generation
- Add error categorization
- Improve meta-prompt with categorized feedback
- **Goal:** High-quality gradient signals

### Phase 3: Robustness
- Add state management and checkpointing
- Implement stopping criteria
- Add progress reporting
- Error handling and retries
- **Goal:** Production-ready reliability

### Phase 4: Polish
- Rich CLI output with progress bars
- Configuration validation
- Example use cases
- Documentation
- **Goal:** Great developer experience

### Phase 5: API Layer (Future)
- REST API with FastAPI
- Async job queue
- Webhooks for completion
- Multi-user support
- **Goal:** Service deployment

## Open Questions for Stakeholder

1. **Target Model Preference:** Should we optimize for a specific model family (e.g., Claude, GPT) or stay model-agnostic?

2. **Eval Format:** Do you want to support additional eval formats beyond input/output pairs and assertions (e.g., LLM-as-judge)?

3. **Observability:** Do you need detailed logging/tracing (e.g., saving all LLM calls for debugging)?

4. **Cost Controls:** Should we add budget limits (max tokens, max cost) as stopping criteria?

5. **Parallel Execution:** Should we run eval cases in parallel for faster feedback, or sequentially for simpler debugging?

## Success Metrics

### Primary Metrics
- **Convergence Rate:** % of tasks that reach 100% eval pass
- **Iteration Count:** Average iterations to convergence
- **Quality:** Final prompt performance on held-out test set

### Secondary Metrics
- **Cost:** Total tokens used per optimization run
- **Time:** Wall-clock time to convergence
- **Stability:** Variance in results across multiple runs

### Evaluation Plan
- Test on 5-10 diverse tasks (sentiment, summarization, extraction, generation, etc.)
- Compare against baseline (no optimization)
- Track all metrics across tasks
- Analyze failure modes for non-convergent cases

## Next Steps

1. **Stakeholder Decision:** Select technology stack approach (A, B, C, or D)
2. **Environment Setup:** Create project structure, dependencies
3. **Prototype Core Loop:** Implement Phase 1 MVP
4. **Validate on Simple Task:** Test with 1-2 eval cases
5. **Iterate:** Add feedback mechanisms, refine meta-prompt
6. **Scale Testing:** Run on multiple diverse tasks
7. **Production Hardening:** Add Phase 3 robustness features

---

**Status:** Awaiting stakeholder decision on technology stack approach before proceeding to implementation.
