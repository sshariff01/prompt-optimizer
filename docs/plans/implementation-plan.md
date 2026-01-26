# Prompt Optimizer Implementation Plan

**Project:** Iterative Meta-Prompt Refinement System
**Approach:** Option A - Custom Implementation (LangSmith integration deferred)
**Date:** 2026-01-25

---

## Executive Summary

Build a prompt optimization system that uses Claude Opus 4.5 to iteratively refine zero-shot prompts based on training/test set performance. The system must reach 100% pass rate on both training and test sets (within cost limits) while avoiding overfitting through descriptive test feedback.

---

## Key Design Decisions

### Architecture
- **Approach:** Iterative Meta-Prompt Refinement (Approach 1)
- **Optimizer:** Claude Opus 4.5 (highest quality meta-reasoning)
- **Target Model:** Configurable (Claude Sonnet, GPT-4, etc.)
- **Prompt Style:** Zero-shot instructions only (no few-shot examples embedded)

### Data Strategy
- **Training Set:** Input/output pairs for optimization with full feedback
- **Test Set:** Held-out validation with descriptive (not specific) feedback
- **Goal:** 100% on both sets without overfitting

### Technology Stack
- **Language:** Python 3.10+
- **LLM SDK:** `anthropic` (official SDK)
- **CLI:** `typer` (modern, type-safe)
- **Config:** `pydantic` + `pydantic-settings` + `yaml`
- **Diffing:** `deepdiff` + `difflib`
- **Output:** `rich` (beautiful CLI)
- **Testing:** `pytest` + `pytest-asyncio`
- **No frameworks:** No DSPy, no LangChain (build from scratch), will add
	LangSmith later

### Model Extensibility Design
**Key Principle:** Implementation designed for easy model swapping

- **Provider Abstraction:** Abstract `LLMProvider` interface separates model calls from core logic
- **Configuration-Driven:** Model selection via YAML config (`optimizer.model`, `target_model.provider`), not hardcoded
- **Default Implementation:**
  - Optimizer: Claude Opus 4.5 (highest quality meta-reasoning)
  - Target: User-configurable (Claude Sonnet, GPT-4, Gemini, etc.)
- **Future Extensions:** Adding new providers (OpenAI, Google, Cohere) requires implementing provider interface, no changes to core components
- **Swap Without Refactoring:** Change config to switch from Claude to GPT-4 without touching code

**Implementation Strategy:**
- `src/optimizer/meta_optimizer.py` depends on `LLMProvider` interface, not `anthropic` SDK directly
- `src/evaluator/test_runner.py` uses same abstraction for target model
- Provider implementations in `src/providers/` (anthropic.py, openai.py, etc.)

### Cost Controls
- Max 30 iterations (~$15-20 budget)
- Token budget: 1M optimizer tokens
- Plateau detection: Stop if no improvement for 7 iterations
- Return best prompt if limits hit

---

## System Architecture

### Components

**1. Optimizer Component** (`src/optimizer/meta_optimizer.py`)
- Uses configurable LLM for meta-prompt engineering (default: Claude Opus 4.5)
- Depends on `LLMProvider` interface, not specific SDK
- Receives comprehensive feedback from evaluations
- Generates/refines zero-shot instruction prompts
- Maintains conversation history for context

**2. Test Runner Component** (`src/evaluator/test_runner.py`)
- Executes prompts against eval cases
- Supports input/output pairs and programmatic assertions
- Configurable target model
- Returns pass/fail + actual outputs

**3. Feedback Analyzer Component** (`src/evaluator/feedback_analyzer.py`)
- **Training feedback:** Full details (inputs, outputs, diffs)
- **Test feedback:** Descriptive patterns without specific examples
- Error categorization (format, boundary, logic, etc.)
- Pattern detection across failures

**4. Orchestrator Component** (`src/orchestrator/optimization_loop.py`)
- Controls two-phase optimization workflow:
  - Phase 1: Train until 100% with full feedback
  - Phase 2: Test validation with descriptive feedback, iterate until 100%
- Implements stopping criteria
- State management and checkpointing

---

## Workflow

### Phase 1: Training Optimization

```
Input: Task description + Training set

Loop until training 100% or limits hit:
  1. Optimizer generates/refines prompt
  2. Test runner evaluates on ALL training cases
  3. Feedback analyzer provides FULL details:
     - Actual input/output pairs
     - Character-level diffs
     - Error categories
     - Root cause analysis
  4. Optimizer refines based on patterns
```

### Phase 2: Test Validation

```
Input: Optimized prompt from Phase 1 + Test set

Loop until test 100% or limits hit:
  1. Test runner evaluates on test set
  2. Feedback analyzer provides DESCRIPTIVE patterns:
     - Error type counts
     - Generic pattern descriptions
     - Root cause hypotheses
     - Recommended fixes
     - NO actual test inputs/outputs
  3. Optimizer refines based on patterns (not memorization)
```

### Success Criteria
- Primary: Training 100% AND Test 100%
- With limits: Best prompt within 30 iterations / 1M tokens
- Quality: Test feedback prevents overfitting

---

## Project Structure

```
prompt-optimizer/
├── src/
│   ├── __init__.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                 # LLMProvider interface
│   │   ├── anthropic.py            # Anthropic implementation
│   │   └── openai.py               # OpenAI implementation (future)
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── meta_optimizer.py       # Model-agnostic prompt refiner
│   │   ├── prompts.py              # Meta-prompt templates
│   │   └── models.py               # Pydantic models
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── test_runner.py          # Execute prompts against evals
│   │   ├── feedback_analyzer.py    # Diff generation + error categorization
│   │   └── eval_types.py           # Input/output + assertion types
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── optimization_loop.py    # Main workflow controller
│   │   ├── state_manager.py        # Checkpointing and persistence
│   │   └── stopping_criteria.py    # Convergence detection
│   └── cli/
│       ├── __init__.py
│       ├── main.py                 # CLI entry point
│       └── commands.py             # CLI commands
├── tests/
│   ├── unit/
│   │   ├── test_meta_optimizer.py
│   │   ├── test_feedback_analyzer.py
│   │   └── test_stopping_criteria.py
│   ├── integration/
│   │   └── test_full_optimization.py
│   └── fixtures/
│       └── sample_evals.yaml
├── examples/
│   ├── sentiment_analysis/
│   │   ├── config.yaml
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   └── code_generation/
│       ├── config.yaml
│       ├── train.jsonl
│       └── test.jsonl
├── docs/
│   └── plans/                      # Design documents (already created)
├── pyproject.toml
├── README.md
└── .env.example
```

---

## Dependencies

### Core (pyproject.toml)

```toml
[project]
name = "prompt-optimizer"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "anthropic>=0.40.0",
    "typer>=0.12.0",
    "pydantic>=2.8.0",
    "pydantic-settings>=2.4.0",
    "pyyaml>=6.0.1",
    "rich>=13.7.0",
    "deepdiff>=7.0.1",
    "jinja2>=3.1.4",
    "aiofiles>=24.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.3.0",
]
```

---

## Configuration Format

### config.yaml

```yaml
task_description: |
  Classify sentiment of product reviews as positive, negative, or neutral.

target_model:
  provider: anthropic
  model: claude-sonnet-4.5
  temperature: 0.0
  max_tokens: 500

optimizer:
  model: claude-opus-4.5
  max_iterations: 30
  max_optimizer_tokens: 1000000
  max_test_iterations: 15
  plateau_threshold: 7

data:
  training_set: ./data/train.jsonl
  test_set: ./data/test.jsonl

feedback:
  training_detail_level: full
  test_detail_level: descriptive

stopping_criteria:
  training_pass_rate: 1.0
  test_pass_rate: 1.0
  on_incomplete_convergence: return_best
```

### train.jsonl / test.jsonl

```jsonl
{"input": "This product is amazing!", "expected_output": "positive"}
{"input": "Broke after one day", "expected_output": "negative"}
{"input": "It's okay, nothing special", "expected_output": "neutral"}
```

---

## Implementation Phases

### Phase 1: Core Loop (MVP)
**Goal:** End-to-end working system

**Tasks:**
1. Project setup (pyproject.toml, directory structure)
2. Basic Pydantic models for config and data
3. Simple meta_optimizer.py (Opus integration)
4. Basic test_runner.py (execute prompt, return pass/fail)
5. Simple orchestrator (loop until training passes)
6. CLI entry point with basic output

**Success Criteria:**
- Can load config + training data
- Can optimize prompt to pass training set
- Prints results to console

**Estimated:** ~500 LOC

---

### Phase 2: Rich Feedback
**Goal:** High-quality gradient signals

**Tasks:**
1. Implement feedback_analyzer.py:
   - Full training feedback (diffs, categories)
   - Descriptive test feedback (patterns, no examples)
2. Error categorization logic:
   - Format violations
   - Boundary confusion
   - Logic errors
   - Pattern detection across failures
3. Enhanced meta-prompt template:
   - Consumes rich feedback
   - Generates targeted refinements
4. Add test set validation phase

**Success Criteria:**
- Training feedback includes character diffs
- Test feedback describes patterns without revealing cases
- Optimizer uses feedback to make intelligent refinements

**Estimated:** ~800 LOC

---

### Phase 3: Robustness
**Goal:** Production-ready reliability

**Tasks:**
1. State management:
   - Save iteration history
   - Checkpoint resume capability
2. Stopping criteria:
   - Max iterations
   - Token budget
   - Plateau detection
3. Error handling:
   - API failures with retries
   - Malformed outputs
   - Timeout handling
4. Progress reporting:
   - Real-time iteration updates
   - Cost tracking

**Success Criteria:**
- Can resume from checkpoint
- Stops gracefully when limits hit
- Clear error messages for failures

**Estimated:** ~600 LOC

---

### Phase 4: Polish
**Goal:** Great developer experience

**Tasks:**
1. Rich CLI output:
   - Progress bars
   - Colored status indicators
   - Tables for results
2. Config validation:
   - Pydantic settings validation
   - Helpful error messages
3. Example use cases:
   - Sentiment analysis
   - Code generation
4. Documentation:
   - README with quickstart
   - API documentation
   - Example configs

**Success Criteria:**
- Beautiful terminal UI
- Clear documentation
- Working examples

**Estimated:** ~400 LOC

---

### Phase 5: LangSmith Integration (Future)
**Deferred:** Add after core system validated

**Tasks:**
1. Add langsmith SDK
2. Wrap Anthropic client for tracing
3. Log eval results to datasets
4. Optional observability layer

---

## Critical Files to Create

### Priority 1 (Phase 1 - MVP)

1. **pyproject.toml** - Dependencies and project metadata
2. **src/providers/base.py** - LLMProvider interface (abstraction for model swapping)
3. **src/providers/anthropic.py** - Anthropic provider implementation
4. **src/optimizer/models.py** - Pydantic models for config, data, results
5. **src/optimizer/meta_optimizer.py** - Model-agnostic optimizer using LLMProvider
6. **src/evaluator/test_runner.py** - Eval execution logic
7. **src/orchestrator/optimization_loop.py** - Main workflow
8. **src/cli/main.py** - CLI entry point
9. **.env.example** - API key template

### Priority 2 (Phase 2 - Rich Feedback)

8. **src/evaluator/feedback_analyzer.py** - Feedback generation
9. **src/optimizer/prompts.py** - Meta-prompt templates
10. **src/evaluator/eval_types.py** - Eval case types

### Priority 3 (Phase 3 - Robustness)

11. **src/orchestrator/state_manager.py** - Checkpointing
12. **src/orchestrator/stopping_criteria.py** - Convergence logic

---

## Key Implementation Details

### Meta-Prompt Template Structure

```python
# src/optimizer/prompts.py

INITIAL_PROMPT_GENERATION = """
You are a prompt engineering expert. Your task is to generate a zero-shot
instruction prompt for the following task:

{task_description}

Here are example input/output pairs showing the desired behavior:
{training_examples}

Generate a clear, concise zero-shot prompt that will perform this task
correctly. The prompt should include:
- Clear instructions
- Format specifications
- Guidance for edge cases

Output ONLY the prompt text, no explanations.
"""

REFINEMENT_PROMPT = """
You are refining a prompt that is not performing perfectly.

Current Prompt:
{current_prompt}

Evaluation Results:
{feedback}

Your task: Analyze the failures and generate an improved version of the prompt.
Focus on addressing the specific error patterns shown in the feedback.

Output ONLY the refined prompt text, no explanations.
"""
```

### Feedback Analyzer - Training vs Test

```python
# src/evaluator/feedback_analyzer.py

class FeedbackAnalyzer:
    def analyze_training_failures(self, failures):
        """Full details for training set"""
        return [
            {
                "input": case.input,
                "expected": case.expected_output,
                "actual": case.actual_output,
                "diff": self._generate_diff(case.expected, case.actual),
                "error_category": self._categorize_error(case),
                "analysis": self._analyze_root_cause(case)
            }
            for case in failures
        ]

    def analyze_test_failures(self, failures):
        """Descriptive patterns without specifics"""
        categorized = self._group_by_error_type(failures)

        return [
            {
                "error_type": error_type,
                "count": len(cases),
                "pattern_observed": self._extract_pattern(cases),
                "generic_example": self._create_generic_example(cases),
                "root_cause": self._hypothesize_cause(cases),
                "recommended_fix": self._suggest_fix(cases)
            }
            for error_type, cases in categorized.items()
        ]
```

### Stopping Criteria Logic

```python
# src/orchestrator/stopping_criteria.py

class StoppingCriteria:
    def should_stop(self, state: OptimizationState) -> tuple[bool, str]:
        """Returns (should_stop, reason)"""

        # Success
        if state.training_pass_rate == 1.0 and state.test_pass_rate == 1.0:
            return True, "success"

        # Hard limits
        if state.iteration >= self.max_iterations:
            return True, "max_iterations"

        if state.optimizer_tokens >= self.max_tokens:
            return True, "token_budget_exceeded"

        if state.test_phase and state.test_iterations >= self.max_test_iterations:
            return True, "test_iteration_limit"

        # Plateau detection
        if self._is_plateaued(state.history):
            return True, "plateau_detected"

        return False, ""

    def _is_plateaued(self, history: list) -> bool:
        """No improvement for N iterations"""
        if len(history) < self.plateau_threshold:
            return False

        recent = history[-self.plateau_threshold:]
        scores = [h.combined_score for h in recent]

        # Check if all scores within minimum_improvement of each other
        return max(scores) - min(scores) < self.minimum_improvement
```

---

## Verification Plan

### Unit Tests

```python
# tests/unit/test_feedback_analyzer.py

def test_training_feedback_includes_full_details():
    analyzer = FeedbackAnalyzer()
    failures = [create_failure_case()]

    feedback = analyzer.analyze_training_failures(failures)

    assert "input" in feedback[0]
    assert "expected" in feedback[0]
    assert "actual" in feedback[0]
    assert "diff" in feedback[0]

def test_test_feedback_excludes_specific_examples():
    analyzer = FeedbackAnalyzer()
    failures = [create_failure_case()]

    feedback = analyzer.analyze_test_failures(failures)

    assert "input" not in feedback[0]  # Should not reveal
    assert "pattern_observed" in feedback[0]  # Should have patterns
```

### Integration Test

```python
# tests/integration/test_full_optimization.py

def test_sentiment_optimization_reaches_100_percent():
    """End-to-end test with small dataset"""
    config = load_config("examples/sentiment_analysis/config.yaml")

    optimizer = PromptOptimizer(config)
    result = optimizer.run()

    assert result.status == "success"
    assert result.training_pass_rate == 1.0
    assert result.test_pass_rate == 1.0
    assert result.iterations <= 30
```

### Manual Verification

1. **Run on sentiment analysis example:**
   ```bash
   python -m src.cli.main optimize examples/sentiment_analysis/config.yaml
   ```
   - Should converge to 100% on both train/test
   - Should complete within 30 iterations
   - Cost should be ~$10-20

2. **Check cost tracking:**
   - Verify token counts are accurate
   - Verify cost estimates match actual API usage

3. **Test stopping criteria:**
   - Create config with max_iterations=5
   - Should stop gracefully and return best prompt
   - Should provide diagnostic information

4. **Test checkpoint resume:**
   - Start optimization
   - Kill process mid-run
   - Resume should continue from checkpoint

---

## Success Metrics

### Functional Requirements
✅ Reaches 100% on training set
✅ Reaches 100% on test set (or best within limits)
✅ Test feedback doesn't reveal specific examples
✅ Stops within cost limits (30 iterations / 1M tokens)

### Quality Metrics
- Typical convergence: 8-15 iterations
- Cost per optimization: $10-20
- Time to completion: 5-15 minutes (depending on eval set size)

### Code Quality
- Test coverage >80%
- Type hints on all public functions
- Clear error messages
- Documented configuration options

---

## Risks & Mitigations

### Risk 1: Test set never reaches 100%
**Mitigation:**
- Descriptive feedback should be "a little bit descriptive"
- Plateau detection stops gracefully
- Return best prompt with diagnostics

### Risk 2: Cost overruns
**Mitigation:**
- Hard limits on iterations and tokens
- Progress tracking shows costs in real-time
- Fail-fast if budget exceeded

### Risk 3: Opus doesn't understand descriptive feedback
**Mitigation:**
- Carefully designed feedback format with examples
- Generic patterns should be actionable
- Future: Can increase specificity if needed

### Risk 4: Training/test distribution mismatch
**Mitigation:**
- User responsible for data quality
- Diagnostics show which patterns resist optimization
- Recommend adding training examples

---

## Next Steps

1. **Phase 1 Implementation:**
   - Set up project structure
   - Implement core optimization loop
   - Validate with simple example

2. **Phase 2 Enhancement:**
   - Add rich feedback mechanisms
   - Test with sentiment analysis example

3. **Phase 3 Hardening:**
   - Add cost controls and stopping criteria
   - Comprehensive error handling

4. **Phase 4 Polish:**
   - Beautiful CLI output
   - Documentation and examples

5. **Future (Phase 5):**
   - LangSmith integration for observability
   - Additional eval types (LLM-as-judge)
   - Multi-model optimization

---

## Related Documents

- [System Design](./2026-01-25-prompt-optimization-system-design.md) - Overall architecture and approach comparison
- [Deep Analysis: DSPy vs LangSmith](./2026-01-25-option-b-d-deep-analysis.md) - Technology stack tradeoff analysis
- [Training Set Workflow](./2026-01-25-training-set-workflow.md) - Data handling and workflow details
- [Test Feedback Strategy](./2026-01-25-test-feedback-strategy.md) - Avoiding overfitting while reaching 100%

---

**Plan Status:** Ready for Execution
**Estimated Total LOC:** ~2000-2500
**Estimated Time:** 2-3 weeks for Phases 1-4
**Estimated Cost per Optimization Run:** $10-20
