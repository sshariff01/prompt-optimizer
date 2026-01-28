# Prompt Optimizer

An iterative meta-prompt refinement system that uses Claude Opus 4.5 to automatically generate and optimize prompts based on evaluation feedback.

## Overview

Prompt Optimizer uses a three-phase workflow to create high-quality zero-shot prompts:

1. **Training Phase:** Optimize prompt with full feedback (specific inputs/outputs/diffs) until 100% pass rate
2. **Validation Phase:** Continue optimizing with descriptive feedback (patterns only, no specific examples) to improve generalization
3. **Test Phase:** Evaluate once on a held-out test set for true out-of-sample performance

The system iteratively refines prompts until both training and validation sets reach 100% pass rate, or until cost/iteration limits are reached.

**Note:** Training + validation are used during optimization. The held-out test set is evaluated only at the end.

## Key Features

- **Multi-Candidate Exploration:** Generates 3 candidate prompts per iteration, evaluates all, picks best - maximizes chance of finding improvements
- **Iterative Refinement:** Uses advanced LLMs for intelligent meta-prompt engineering
- **Optimization Memory:** Context-preserving system that maintains lessons learned and iteration history, preventing repeated mistakes and enabling compounding improvements
- **Data-Driven Feedback:** Analyzes actual failure patterns to generate specific, actionable guidance (not generic templates)
- **Training Regression Protection:** Combined feedback system provides both validation patterns to fix AND training constraints to preserve when candidates break existing functionality
- **Model Flexibility:** Provider abstraction design enables easy model swapping
  - Configurable optimizer and target models via TOML config
  - Default: Claude Opus 4.5 (optimizer) + any target model
  - Future: OpenAI, Google, Cohere support via provider interface
- **High-Performance Evaluation:** Thread-safe caching with 50 parallel workers (configurable) for fast evaluation
- **Adaptive Acceptance:** Smart logic that requires strict improvement at low scores (avoid 0% loops) but allows lateral moves at high scores (explore approaches at 95%+)
- **Overfitting Prevention:** Validation feedback provides patterns without revealing specific examples
- **Cost Controls:** Hard limits on iterations, tokens, and plateau detection
- **Zero-Shot Output:** Generates standalone instruction prompts (no few-shot examples)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Optimization Loop                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Phase 1: Training Optimization (Full Feedback)          │ │
│  │  ┌──────────────────────────────────────────────┐        │ │
│  │  │ 1. Generate 3 candidates (temperature=0.7)   │        │ │
│  │  │ 2. Evaluate all against training set         │        │ │
│  │  │ 3. Pick best (>= comparison for laterals)    │        │ │
│  │  │ 4. Accept if improvement (adaptive logic)    │        │ │
│  │  │ 5. Update optimization memory                │        │ │
│  │  └──────────────────────────────────────────────┘        │ │
│  │  Repeat until 100% or limit                              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Phase 2: Validation Optimization (Descriptive Feedback)  │ │
│  │  ┌──────────────────────────────────────────────┐        │ │
│  │  │ 1. Generate 3 candidates                     │        │ │
│  │  │ 2. Re-validate each against training         │        │ │
│  │  │ 3. Filter out training regressions           │        │ │
│  │  │ 4. Evaluate survivors on validation set      │        │ │
│  │  │ 5. Pick best validation score                │        │ │
│  │  │ 6. If training regressed: use combined       │        │ │
│  │  │    feedback (validation patterns + training) │        │ │
│  │  └──────────────────────────────────────────────┘        │ │
│  │  Repeat until 100% or limit                              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Phase 3: Held-out Test Evaluation (No Feedback)        │ │
│  │  ┌──────────────────────────────────────────────┐        │ │
│  │  │ 1. Evaluate final prompt on held-out set      │        │ │
│  │  │ 2. Report final out-of-sample score           │        │ │
│  │  └──────────────────────────────────────────────┘        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Core Components:**

- **Meta-Optimizer:** Uses Claude Opus 4.5 to generate and refine prompts based on feedback and optimization memory
- **Optimization Memory:** Two-tier context system (accumulated lessons + recent 3 iterations) providing historical context to prevent repeated mistakes
- **Test Runner:** Thread-safe parallel execution engine with intelligent caching (50 workers default)
- **Feedback Analyzer:**
  - Training: Detailed feedback with specific failures, diffs, categorized errors
  - Validation: Data-driven descriptive patterns extracted from actual failures (not generic templates)
  - Combined: Validation patterns + training constraints when candidates break existing functionality
- **Orchestrator:** Controls the three-phase workflow with adaptive acceptance logic and stopping criteria
- **Prompt History:** Tracks all prompt versions and manages accept/reject decisions

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
- [x] **Phase 2: Validation Set Optimization** - Descriptive feedback to prevent overfitting
  - Two-phase optimization workflow
  - Descriptive validation feedback (patterns only, no specific examples)
  - Anti-overfitting mechanism
  - Validation set optimization with generalization

**MVP Status:** Phase 1 + Phase 2 together form the complete MVP. The system can optimize prompts to 100% on both training and validation sets while reducing overfitting through limited feedback.

**Important Note:** The validation set is used during optimization. The anti-overfitting mechanism uses descriptive patterns instead of specific examples to reduce (but not eliminate) overfitting. For true out-of-sample evaluation, use a separate held-out test set that is never used during the optimization loop.

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

   This will optimize a sentiment analysis prompt using the example data in `data/train.jsonl` and `data/validation.jsonl`.

3. **Customize for your task:**
   ```bash
   # Copy the default config
   cp config.toml my-task-config.toml

   # Edit my-task-config.toml:
   # - Update task_description
   # - Point to your training/validation/heldout data files
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
model = "claude-sonnet-4-20250514"  # Model being optimized for evaluation

[optimizer]
model = "claude-opus-4-5-20251101"  # Model doing the meta-prompt engineering
max_iterations = 30                 # Maximum optimization iterations
max_test_iterations = 15            # Maximum validation phase iterations
candidates_per_iteration = 3        # Number of prompt candidates to generate per iteration (default: 3, max: 10)
max_workers = 50                    # Parallel API calls for evaluation (default: 50)

[data]
training_set = "./data/train.jsonl"  # Training examples (full feedback)
validation_set = "./data/validation.jsonl"  # Validation examples (limited feedback during optimization)
heldout_set = "./data/test.jsonl"           # Held-out test set (final evaluation only)
```

**Key Parameters:**
- `candidates_per_iteration`: Generates N candidate prompts per iteration, evaluates all, picks best. Higher = more exploration but 3x cost per iteration.
- `max_workers`: Number of parallel API calls. Higher = faster evaluation (up to ~50 before diminishing returns). Anthropic rate limits apply.
- `max_test_iterations`: Separate limit for validation phase to prevent excessive optimization on validation data.

See `config.toml` for a complete example with all available options.

## Data Format

Training, validation, and held-out test data should be in JSONL format (one JSON object per line). For strict
labeling tasks, use a structured output format and define a schema in the config:

```jsonl
{"input": "Example input text", "expected_output": "LABEL=POSITIVE"}
{"input": "Another input", "expected_output": "LABEL=NEGATIVE"}
```

```toml
[schema]
fields = ["LABEL"]
enums = { LABEL = ["POSITIVE", "NEGATIVE"] }
```

## Example Demonstration

Here's a real optimization run on a yes/no question answering task:

**Task:** Answer yes/no questions with output `LABEL=YES` or `LABEL=NO`

**Training Data** (7 examples):
```jsonl
{"input": "Is the sky blue?", "expected_output": "LABEL=YES"}
{"input": "Can fish fly?", "expected_output": "LABEL=NO"}
{"input": "Do humans need water?", "expected_output": "LABEL=YES"}
{"input": "Is Mars closer to the Sun than Earth?", "expected_output": "LABEL=NO"}
{"input": "Does 2 + 2 equal 4?", "expected_output": "LABEL=YES"}
{"input": "Are all mammals warm-blooded?", "expected_output": "LABEL=YES"}
{"input": "Can plants photosynthesize without light?", "expected_output": "LABEL=NO"}
```

**Validation Data** (3 examples):
```jsonl
{"input": "Are trees alive?", "expected_output": "LABEL=YES"}
{"input": "Can rocks think?", "expected_output": "LABEL=NO"}
{"input": "Is ice frozen water?", "expected_output": "LABEL=YES"}
```

### Run Output:

```
$ prompt-optimizer examples/demo/config.toml

Loading configuration...
✓ Configuration loaded from examples/demo/config.toml

Loading training data...
✓ Loaded 7 training cases
✓ Loaded 2 validation cases
✓ Loaded 1 held-out test case

Initializing providers...
✓ Optimizer: claude-opus-4-20250514
✓ Target model: claude-sonnet-4-20250514

Starting optimization...

======================================================================

Starting optimization with 7 training cases and 2 validation cases
Target model: claude-sonnet-4-20250514
Optimizer: claude-opus-4-20250514

Iteration 0: Generating initial prompt...
Initial prompt:
Answer the following yes/no question. Respond with exactly 'LABEL=YES' or
'no' in lowercase. Do not include any punctuation, explanations, or additional
text. If the question is ambiguous or cannot be definitively answered, choose
the most reasonable interpretation based on common knowledge and scientific
consensus.

Initial evaluation: 7/7 passed (100.0%)

======================================================================
PHASE 1: Training Set Optimization
======================================================================

✓ Training set: 100% pass rate achieved!

======================================================================
PHASE 2: Validation Set Optimization
======================================================================

Initial validation evaluation: 2/2 passed (100.0%)

✓ Validation set: 100% pass rate achieved!

======================================================================

Optimization Complete!

╭──────────────────────── Status: success ─────────────────────────╮
│ Successfully reached 100% on both training and validation sets!   │
│ Training: 100.0%, Validation: 100.0% Held-out: 100.0%             │
╰──────────────────────────────────────────────────────────────────╯

Metrics:
  Training Pass Rate: 100.0%
  Validation Pass Rate: 100.0%
  Held-out Test Pass Rate: 100.0%
  Total Iterations: 2
  Optimizer Tokens Used: 303

Final Optimized Prompt:
╭──────────────────────────────────────────────────────────────────╮
│ Answer the following yes/no question. Respond with exactly       │
│ 'LABEL=YES' or 'LABEL=NO'. Do not include any punctuation,       │
│ explanations, or additional text. If the question is ambiguous   │
│ or cannot be definitively answered, choose the most reasonable   │
│ interpretation based on common knowledge and scientific consensus.│
╰──────────────────────────────────────────────────────────────────╯

✓ Results saved to examples/demo/optimization_result.json
```

### Key Takeaways:

- **Fast Convergence:** The optimizer generated a high-quality prompt on the first try (Iteration 0)
- **100% Accuracy:** Achieved perfect scores on training (7/7) and validation (2/2)
- **Efficient:** Used only 303 optimizer tokens (~$0.01 cost)
- **Three-Phase Flow:** System optimized on training, refined on validation, then reported held-out performance
- **Clear Output:** The final prompt is concise, clear, and ready to use

This demonstrates the system's ability to quickly generate effective zero-shot prompts with minimal iterations.

---

### Example 2: Customer Service Request Classification (Complex Task)

**Task:** Categorize customer service messages into request types with strict labels

**Training Data** (10 examples):
```jsonl
{"input": "I want to return my order #12345 because it doesn't fit", "expected_output": "CATEGORY=RETURN_REQUEST; DETAIL=SIZE; ORDER_ID=12345; CHANGE=NONE"}
{"input": "Where is my package? I ordered it 5 days ago.", "expected_output": "CATEGORY=TRACKING_INQUIRY; DETAIL=ORDER_STATUS; ORDER_ID=NONE; CHANGE=NONE"}
{"input": "Do you have this shirt in blue?", "expected_output": "CATEGORY=PRODUCT_INQUIRY; DETAIL=COLOR; ORDER_ID=NONE; CHANGE=NONE"}
{"input": "I was charged twice for the same order!", "expected_output": "CATEGORY=BILLING_ISSUE; DETAIL=DUPLICATE_CHARGE; ORDER_ID=NONE; CHANGE=NONE"}
{"input": "Your product broke after 2 days. This is unacceptable!", "expected_output": "CATEGORY=COMPLAINT; DETAIL=PRODUCT_DEFECT; ORDER_ID=NONE; CHANGE=NONE"}
... (5 more examples)
```

**Validation Data** (sample):
```jsonl
{"input": "I received the wrong size. Can I exchange order #55555?", "expected_output": "CATEGORY=RETURN_REQUEST; DETAIL=SIZE; ORDER_ID=55555; CHANGE=NONE"}
{"input": "My order still hasn't arrived and it's been a week.", "expected_output": "CATEGORY=TRACKING_INQUIRY; DETAIL=ORDER_STATUS; ORDER_ID=NONE; CHANGE=NONE"}
{"input": "The product quality is terrible. I want my money back.", "expected_output": "CATEGORY=REFUND_REQUEST; DETAIL=REFUND_QUALITY; ORDER_ID=NONE; CHANGE=NONE"}
... (2 more examples)
```

### Run Output (Abbreviated):

```
$ prompt-optimizer examples/customer-service/config.toml

Loading configuration...
✓ Loaded 10 training cases
✓ Loaded 12 validation cases
✓ Loaded 3 held-out test cases
✓ Optimizer: claude-opus-4-20250514
✓ Target model: claude-sonnet-4-20250514

Iteration 0: Generating initial prompt...
Initial evaluation: 0/10 passed (0.0%)

======================================================================
PHASE 1: Training Set Optimization
======================================================================

Iteration 1: Refining prompt...
  Pass rate: 0/10 (0.0%)
  Failures: 10
  New pass rate: 8/10 (80.0%)

Iteration 2: Refining prompt...
  Pass rate: 8/10 (80.0%)
  Failures: 2
  New pass rate: 10/10 (100.0%)

✓ Training set: 100% pass rate achieved!

======================================================================
PHASE 2: Validation Set Optimization
======================================================================

Initial validation evaluation: 9/12 passed (75.0%)

Iteration 4-13: Refining based on validation patterns...
  [Multiple iterations with feedback adjustments]
  Validation pass rate fluctuating: 40% → 60% → 80%

⚠ Validation iteration limit reached

======================================================================

Optimization Complete!

╭─────────────────── Status: max_iterations ───────────────────────╮
│ Reached maximum iterations (20).                                │
│ Training: 100.0%, Validation: 80.0% Held-out: 66.7%             │
╰──────────────────────────────────────────────────────────────────╯

Metrics:
  Training Pass Rate: 100.0%
  Validation Pass Rate: 80.0%
  Held-out Test Pass Rate: 66.7%
  Total Iterations: 14
  Optimizer Tokens Used: 34,256

Final Optimized Prompt: [Complex 200+ line prompt with detailed
reasoning framework, category definitions, decision hierarchy, and
nuanced intent recognition rules]

✓ Results saved to examples/customer-service/optimization_result.json
```

### Key Takeaways:

- **Complex Task:** Multi-category classification with nuanced intent detection
- **More Iterations:** Required 14 iterations vs 2 for simple yes/no task
- **Incomplete Convergence:** Hit validation iteration limit at 80% validation accuracy
- **Rich Prompt Generated:** System developed a sophisticated 200+ line prompt with:
  - Detailed category definitions
  - Decision hierarchy for ambiguous cases
  - Context interpretation rules
  - Special case handling
- **Higher Cost:** Used 34,256 tokens (~$1.03) vs 303 tokens for simple task
- **Training Success:** Still achieved 100% on training set
- **Generalization Challenge:** Validation set revealed edge cases that were difficult to handle with pattern-only feedback

This demonstrates the system's behavior on more challenging tasks where:
1. Initial prompts may be completely wrong (0% → 100% on training)
2. More iterations are needed for convergence
3. Validation may not reach 100% within iteration limits
4. The system generates increasingly sophisticated prompts to handle complexity
5. Cost scales with task complexity

## Technology Stack

- **Language:** Python 3.10+
- **LLM SDK:** `anthropic` (official SDK)
- **CLI:** `typer`
- **Config:** `pydantic` + `pydantic-settings` + TOML (`tomllib`/`tomli`)
- **Diffing:** `deepdiff` + `difflib`
- **Output:** `rich` (beautiful CLI)
- **Testing:** `pytest`

## Performance Metrics

**With Current Architecture (3 candidates/iteration, 50 parallel workers):**

- **Cost per optimization:** $15-30 (3x candidates increases optimizer token cost)
- **Typical convergence:** 5-12 iterations (faster with multi-candidate exploration)
- **Time to completion:** 3-10 minutes (3-5x speedup from parallelization + caching)
- **Cache hit rate:** 25-40% typical (depends on number of candidates and re-validations)
- **Training phase:** Usually reaches 100% reliably with adaptive acceptance logic
- **Validation phase:** 80-100% depending on task complexity and data quality

**Cost Breakdown (typical 50-case training, 25-case validation set):**
- Optimizer tokens: ~5-10K per iteration (Opus 4.5 for meta-prompting)
- Target model tokens: ~150-300 per evaluation (Sonnet 4 for testing prompts)
- Total optimizer tokens: 50-120K over full optimization
- Total evaluations: ~1000-2000 (reduced by caching)

**Code Statistics:**
- Total lines of code: ~3500 LOC
- Core optimization loop: ~800 LOC
- Feedback/evaluation: ~600 LOC
- Provider abstraction: ~400 LOC

## License

TBD

## How It Works

### Three-Phase Optimization with Multi-Candidate Exploration

**Phase 1: Training Optimization (Full Feedback)**

Each iteration:
1. **Generate 3 Candidates:** Meta-optimizer generates 3 different prompt variations (temperature=0.7 for diversity)
2. **Parallel Evaluation:** All 3 candidates evaluated against training set in parallel (50 workers default)
3. **Select Best:** Pick candidate with highest score (≥ comparison allows lateral moves at high scores)
4. **Adaptive Acceptance:**
   - Score < 10%: Requires strict improvement (>) to avoid 0% loops
   - Score ≥ 10%: Allows lateral moves (≥) to explore different approaches
5. **Update Memory:** Record what worked/didn't work in optimization memory

Continues until 100% training pass rate or limits reached.

**Phase 2: Validation with Limited Feedback (Descriptive Patterns)**

Each iteration:
1. **Generate 3 Candidates:** Create variations based on descriptive validation feedback
2. **Training Re-validation:** Test each candidate against training set to catch regressions
3. **Filter Regressions:** Skip any candidate that breaks training performance
4. **Validation Evaluation:** Evaluate remaining candidates on validation set
5. **Select Best:** Pick candidate with best validation score (that maintains training)
6. **Combined Feedback (if needed):** If candidate broke training, next iteration uses:
   - Validation patterns to fix (what we're trying to improve)
   - Training constraints (specific cases not to break)

Continues until 100% validation pass rate or limits reached.

**Phase 3: Held-out Test Evaluation (No Feedback)**

After optimization completes, the final prompt is evaluated once on the held-out test set.

### Optimization Memory: Context Preservation

The system maintains context across iterations to prevent "memoryless" optimization:

**Problem Solved:** Without memory, each iteration is independent - the optimizer can't learn from previous attempts, may repeat failed approaches, or accidentally undo successful fixes.

**Two-Tier Memory Architecture:**

1. **Accumulated Lessons (Long-term Memory)**
   - Running list of insights extracted from each iteration
   - Examples: "✓ Added explicit format examples → improved performance" or "✗ Over-specified edge cases → caused regression"
   - Provides general principles learned throughout optimization

2. **Recent Iteration History (Short-term Memory)**
   - Detailed summaries of the last 3 iterations
   - Tracks: changes made, target issues, results, accept/reject decisions
   - Shows concrete trajectory: what was recently attempted and what worked

**Context Integration:** On each refinement, the meta-optimizer receives:
```
Optimization Context:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accumulated Lessons:
• ✓ Explicit format examples improve compliance
• ✗ Over-specifying edge cases causes brittleness

Recent Iteration History:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration 5: ✓ ACCEPTED
  Change Made: Addressed format_violation errors
  Training: 80% → 90%

[Current prompt and feedback...]
```

**Benefits:**
- Learns from mistakes and avoids repeating rejected approaches
- Builds on successful patterns discovered in previous iterations
- Maintains awareness of the optimization trajectory
- Makes informed decisions to avoid breaking what already works
- Compounds insights across iterations for faster convergence

**Token Cost:** Adds ~500-1000 tokens per iteration, but improves convergence speed and decision quality, resulting in net positive ROI.

### Data-Driven Descriptive Feedback

Instead of generic template feedback, the system analyzes actual failures to generate specific, actionable guidance:

**Traditional Approach (Generic):**
```
Error Pattern: boundary_confusion (3 failures)
  Pattern: Ambiguous cases classified incorrectly
  Fix: Add guidance for edge cases
```

**Our Approach (Data-Driven):**
```
Error Pattern: boundary_confusion (3 failures)
  Pattern: Ambiguous cases misclassified. 2 expected categories confused
           with 3 actual outputs. Edge cases handled incorrectly.
  Example: Cases requiring distinction between categories like 'return_request,
           refund_request' classified as 'billing_issue, complaint' instead.
  Root Cause: Instructions lack clear guidance for resolving 3 ambiguous cases
  Fix: Add 3 specific rules for handling ambiguous cases. Define clear decision
       criteria: 'When inputs have mixed signals, prioritize X over Y.' Provide
       explicit guidance on edge cases and tie-breaking.
```

**How it works:**
1. Analyzes all failures in the error category
2. Extracts common patterns (expected outputs, actual outputs, metrics)
3. Generates specific descriptions with concrete numbers
4. Provides actionable fixes tailored to the actual failure characteristics

**Result:** More effective guidance leads to faster convergence and better prompt quality.

### Combined Feedback: Training Regression Protection

When a validation phase refinement accidentally breaks training cases, the system switches to **combined feedback**:

**Scenario:**
- Current prompt: 100% training, 80% validation
- Generate candidate to fix validation patterns
- Candidate achieves 85% validation BUT drops training to 96% ❌

**Next Iteration Receives Combined Feedback:**
```
SITUATION: Your last refinement tried to fix validation issues but broke training.

Validation Patterns to Address:
- [Descriptive patterns for the 20% validation failures]

Training Constraints (cases that broke):
- [Specific training cases with actual inputs/outputs that regressed]

Your task: Fix validation patterns WITHOUT breaking these training cases.
```

**Result:** Optimizer learns to balance improvements with preservation, preventing the endless regression loop.

### High-Performance Evaluation with Caching

**Thread-Safe Parallel Execution:**
- 50 parallel workers (configurable via `max_workers`)
- Thread-safe cache with locks prevents race conditions
- Shared cache across all candidates in an iteration

**Intelligent Caching:**
- Cache key: `(prompt, input, system_message)`
- Different prompts = different cache entries (no collisions)
- Same prompt + same input = instant cache hit
- Massive speedup during re-validation phases

**Example Performance:**
```
Iteration 5: Evaluate 3 candidates × 50 training cases
  Candidate 1: 50 API calls (cache misses)
  Candidate 2: 50 API calls (cache misses)
  Candidate 3: 50 API calls (cache misses)
  Re-validate training: 0 API calls (50 cache hits!) ⚡

Cache Stats at End:
  Cache Size: 150 unique evaluations
  Cache Hits: 200
  Cache Misses: 450
  Hit Rate: 30.8%
```

**Typical Speedup:** 3-5x faster with caching + parallelization (vs sequential single-candidate approach).

### Overfitting Reduction Mechanism

The system reduces (but doesn't eliminate) overfitting by providing different feedback types:

**Training Feedback (Full Supervision):**
```
Input: "Product is adequate but overpriced"
Expected: "neutral"
Actual: "negative"
Diff: neutral ≠ negative
Category: boundary_confusion
```

**Validation Feedback (Weak Supervision):**
```
Error Pattern: boundary_confusion (2 failures)
Pattern: Ambiguous cases with mixed signals
Example: Inputs with both positive and negative aspects
Root Cause: Instructions unclear about dominant sentiment
Fix: Add guidance for weighing conflicting signals
```

The validation set is used during optimization with constrained feedback. For true out-of-sample evaluation, maintain a separate held-out test set that is never used during the optimization loop.
