# Prompt Optimizer

An iterative meta-prompt refinement system that uses Claude Opus 4.5 to automatically generate and optimize prompts based on evaluation feedback.

## Overview

Prompt Optimizer uses a two-phase optimization workflow to create high-quality zero-shot prompts:

1. **Training Phase:** Optimize prompt with full feedback until 100% training set pass rate
2. **Test Phase:** Validate on held-out test set with descriptive (non-specific) feedback to avoid overfitting

The system iteratively refines prompts until both training and test sets reach 100% pass rate, or until cost/iteration limits are reached.

## Key Features

- **Iterative Refinement:** Uses advanced LLMs for intelligent meta-prompt engineering
- **Model Flexibility:** Provider abstraction design enables easy model swapping
  - Configurable optimizer and target models via TOML config
  - Default: Claude Opus 4.5 (optimizer) + any target model
  - Future: OpenAI, Google, Cohere support via provider interface
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

✅ **MVP Complete** - Two-phase optimization system is functional

**Completed:**
- [x] Complete system design and architecture documentation
- [x] **Phase 1: Core Loop** - Training optimization with full feedback
  - Provider abstraction layer (LLMProvider interface)
  - Meta-optimizer using Claude Opus 4.5
  - Test runner for evaluation
  - Feedback analyzer with detailed training feedback (diffs, error categories)
  - Optimization loop with stopping criteria
  - CLI with console output
- [x] **Phase 2: Test Set Validation** - Descriptive feedback to prevent overfitting
  - Two-phase optimization workflow
  - Descriptive test feedback (patterns only, no specific examples)
  - Anti-overfitting mechanism
  - Test set validation with generalization

**MVP Status:** Phase 1 + Phase 2 together form the complete MVP. The system can optimize prompts to 100% on both training and test sets while preventing overfitting.

**In Progress:**
- [ ] Phase 3: Robustness - State management, checkpointing, retry logic
- [ ] Phase 4: Polish - Rich CLI UI, progress bars, multiple examples
- [ ] Phase 5: LangSmith integration (future)

## Installation

```bash
# Clone the repository
git clone https://github.com/sshariff01/prompt-optimizer.git
cd prompt-optimizer

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

1. **Set your Anthropic API key:**

   **Option A: Using .env file (recommended)**
   ```bash
   # Copy the example
   cp .env.example .env

   # Edit .env and add your key
   # ANTHROPIC_API_KEY=sk-ant-api03-xxx...your-actual-key
   ```

   **Option B: Environment variable**
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

2. **Run optimization with the default config:**
   ```bash
   prompt-optimizer config.toml
   ```

   This will optimize a sentiment analysis prompt using the example data in `data/train.jsonl` and `data/test.jsonl`.

3. **Customize for your task:**
   ```bash
   # Copy the default config
   cp config.toml my-task-config.toml

   # Edit my-task-config.toml:
   # - Update task_description
   # - Point to your training/test data files
   # - Adjust model settings if needed

   # Run optimization
   prompt-optimizer my-task-config.toml
   ```

## Configuration

The system uses TOML configuration files. Key settings:

```toml
# Task description (what you want the prompt to do)
task_description = "Your task description here"

[target_model]
model = "claude-sonnet-4.5"  # Model being optimized

[optimizer]
model = "claude-opus-4.5"     # Model doing the optimization
max_iterations = 30           # Maximum optimization iterations

[data]
training_set = "./data/train.jsonl"  # Training examples
test_set = "./data/test.jsonl"       # Test examples (held-out)
```

See `config.toml` for a complete example with all available options.

## Data Format

Training and test data should be in JSONL format (one JSON object per line):

```jsonl
{"input": "Example input text", "expected_output": "expected result"}
{"input": "Another input", "expected_output": "another result"}
```

## Technology Stack

- **Language:** Python 3.10+
- **LLM SDK:** `anthropic` (official SDK)
- **CLI:** `typer`
- **Config:** `pydantic` + `pydantic-settings` + TOML (`tomllib`/`tomli`)
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

## How It Works

### Two-Phase Optimization

**Phase 1: Training Optimization**
- Optimizer receives full feedback (actual inputs, outputs, diffs)
- Iteratively refines prompt until 100% training pass rate
- Uses detailed error categorization and root cause analysis

**Phase 2: Test Validation**
- Optimizer receives only descriptive feedback (patterns, not examples)
- Prevents overfitting by hiding specific test cases
- Iteratively refines prompt to generalize to test set
- Continues until 100% test pass rate or limits reached

### Anti-Overfitting Mechanism

**Training Feedback (Full):**
```
Input: "Product is adequate but overpriced"
Expected: "neutral"
Actual: "negative"
Diff: neutral ≠ negative
Category: boundary_confusion
```

**Test Feedback (Descriptive):**
```
Error Pattern: boundary_confusion (2 failures)
Pattern: Ambiguous cases with mixed signals
Example: Inputs with both positive and negative aspects
Root Cause: Instructions unclear about dominant sentiment
Fix: Add guidance for weighing conflicting signals
```

This ensures the optimizer learns to generalize rather than memorize test cases.

## Contributing

The MVP (Phase 1 + Phase 2) is complete and functional. Contributions for Phase 3 (robustness) and Phase 4 (polish) are welcome!
