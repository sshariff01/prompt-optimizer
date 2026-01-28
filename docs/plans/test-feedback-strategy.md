# Validation Feedback Strategy: Avoiding Overfitting While Reaching 100%

**Date:** 2026-01-25
**Critical Requirement:** System must reach 100% on BOTH training and validation sets
**Strategy:** Use descriptive (but not specific) validation feedback to avoid overfitting

**Note:** A separate held-out test set is evaluated only at the end for true out-of-sample performance.

---

## The Challenge

**Requirement:** Achieve 100% pass rate on validation set

**Constraint:** Cannot show validation case examples to optimizer (causes overfitting)

**Solution:** Provide "a little bit descriptive" feedback about validation failures without revealing actual validation inputs/outputs

---

## Three-Phase Workflow

### Phase 1: Training Set Optimization (Full Feedback)

**Data:** Training set only

**Feedback Level:** FULL DETAILS
```
Training Results: 18/20 passed (90%)

Detailed Failures:

Case 7:
  Input: "Product is adequate but overpriced"
  Expected Output: "neutral"
  Actual Output: "negative"

  Diff: neutral ≠ negative

  Error Category: Boundary confusion
  Analysis: Mixed sentiment (positive quality + negative price)
           incorrectly classified as negative. Instruction
           needs clarification about how to handle mixed sentiments.

Case 12:
  Input: "Love it!"
  Expected Output: "positive"
  Actual Output: "positive - customer is very satisfied"

  Diff: "positive" ≠ "positive - customer is very satisfied"

  Error Category: Format violation
  Analysis: Output included explanation. Instruction should
           emphasize outputting ONLY the classification label.
```

**Why Full Details:** Optimizer can see patterns across multiple examples, iterate quickly, understand exact failure modes.

**Stopping Criteria:** 100% training pass rate

---

### Phase 2: Validation Set Optimization (Descriptive But Abstract Feedback)

**Data:** Validation set (used during optimization)

**Feedback Level:** DESCRIPTIVE WITHOUT EXAMPLES

#### Example 1: Good Descriptive Feedback

```
Validation Results: 8/10 passed (80%)

Error Type 1: Format Violation (2 failures)
  Pattern Observed: Output included additional explanatory text
                    beyond the required classification label

  Example Pattern: Instead of outputting just the label, the model
                   added phrases like "because..." or "- reasoning"

  Root Cause: Instructions not explicit enough about output format

  Recommended Fix: Add explicit constraint that output must contain
                   ONLY the classification label with no additional text

Error Type 2: Boundary Confusion (1 failure)
  Pattern Observed: Ambiguous case involving mixed signals was
                    resolved incorrectly

  Example Pattern: Input contained both positive and negative aspects,
                   but classification chose wrong dominant sentiment

  Root Cause: Instructions unclear about how to weigh conflicting
              signals when determining overall sentiment

  Recommended Fix: Add guidance about identifying the dominant
                   sentiment when mixed signals are present
```

**What's Included:**
✅ Error categories and counts
✅ Description of the pattern/behavior observed
✅ Generic example of the type of mistake (not the actual input)
✅ Root cause analysis
✅ Specific guidance for fixing

**What's NOT Included:**
❌ Actual validation case inputs
❌ Expected outputs
❌ Actual outputs
❌ Specific diffs

---

### Phase 3: Held-out Test Evaluation (No Feedback)

**Data:** Held-out validation set (never used during optimization)

**Feedback Level:** NONE (final evaluation only)

**Purpose:** Report out-of-sample performance after training + validation optimization completes.

---

#### Example 2: Too Generic (Not Helpful)

```
❌ BAD FEEDBACK:

Validation Results: 8/10 passed (80%)

- 2 format errors
- 1 boundary error
```

**Why This Fails:** Not actionable. Optimizer doesn't know what to fix.

---

#### Example 3: Too Specific (Causes Overfitting)

```
❌ TOO SPECIFIC:

Validation Results: 8/10 passed (80%)

Case 3:
  Input: "Love it!"
  Expected: "positive"
  Got: "positive - customer is very satisfied"
```

**Why This Fails:** Optimizer sees actual validation data, can memorize it. This IS using validation as training data.

---

## The "Just Right" Balance

### Feedback Template for Test Failures

```
Validation Results: {pass_count}/{total} passed ({percentage}%)

[For each error type:]

Error Type {N}: {Category Name} ({count} failures)

  Pattern Observed: {High-level description of what went wrong}

  Example Pattern: {Generic illustration without actual validation data}
                   "e.g., inputs with mixed signals" NOT
                   "Input: 'Good but expensive' → 'negative'"

  Root Cause: {Why this is happening based on current instructions}

  Recommended Fix: {Specific guidance on how to modify instructions}
```

### Real-World Examples

#### Example A: Sentiment Classification

**Descriptive Test Feedback:**
```
Validation Results: 7/10 passed (70%)

Error Type 1: Sarcasm Detection (2 failures)
  Pattern Observed: Sarcastic statements were interpreted literally
                    and classified with opposite sentiment

  Example Pattern: Phrases like "Oh great, just what I needed..."
                   when describing a problem were classified as
                   positive instead of negative due to the word "great"

  Root Cause: Instructions don't mention sarcasm or provide
              contextual interpretation guidance

  Recommended Fix: Add guidance to identify sarcasm markers:
                   - Positive words in negative contexts
                   - Phrases like "Oh great", "Wonderful", "Perfect"
                     when followed by problem descriptions
                   - Context overrides literal word sentiment

Error Type 2: Neutral Boundary (1 failure)
  Pattern Observed: Factual statement without emotional language
                    was classified as having sentiment

  Example Pattern: Purely descriptive statements about product
                   features without satisfaction indicators

  Root Cause: Instructions may not clearly define neutral category
              as "absence of sentiment" vs "mixed sentiment"

  Recommended Fix: Clarify neutral means:
                   - Factual descriptions only
                   - No satisfaction indicators
                   - No problems or praise mentioned
```

---

#### Example B: Code Generation

**Descriptive Test Feedback:**
```
Validation Results: 9/12 passed (75%)

Error Type 1: Edge Case Handling (2 failures)
  Pattern Observed: Generated code missing validation for boundary
                    conditions

  Example Pattern: Functions didn't handle empty inputs, zero values,
                   or null cases that weren't in training examples

  Root Cause: Instructions focus on happy path, don't emphasize
              defensive programming

  Recommended Fix: Add requirement to:
                   - Always validate inputs
                   - Handle empty/null/zero cases
                   - Include error handling for edge cases

Error Type 2: Documentation Missing (1 failure)
  Pattern Observed: Code generated without inline comments or
                    function docstrings

  Example Pattern: Functions with complex logic had no explanatory
                   comments, making intent unclear

  Root Cause: Instructions don't specify documentation requirements

  Recommended Fix: Require all functions include:
                   - Docstring with purpose, params, returns
                   - Inline comments for non-obvious logic
```

---

## How Opus Uses This Feedback

### Iteration Flow with Descriptive Feedback

**Current Prompt:**
```
Classify sentiment as positive, negative, or neutral.
Output only the label in lowercase.
```

**Test Feedback Received:**
```
Error Type: Sarcasm Detection (2 failures)
  Pattern: Sarcastic positive words in negative contexts
           classified incorrectly
  Fix: Add sarcasm detection guidance
```

**Opus Reasoning:**
"Validation set reveals sarcasm handling issue not apparent in training.
Need to add guidance about contextual interpretation without
overfitting to specific sarcastic phrases."

**Refined Prompt:**
```
Classify sentiment as positive, negative, or neutral.

Important: Consider context when interpreting language.
- Sarcasm: Positive words in negative contexts indicate negative sentiment
  Example patterns: "Oh great..." "Wonderful..." "Perfect..." + complaint
- Sincere: Positive words in positive contexts indicate positive sentiment
- Factual: No emotional language indicates neutral

Output only the label in lowercase.
```

**Key:** Opus adds *generalized* guidance about sarcasm, not specific phrases from validation cases.

---

## Why This Approach Works

### Advantages

✅ **Prevents Overfitting**
- Optimizer never sees validation case specifics
- Can't memorize validation inputs/outputs
- Forces generalization

✅ **Actionable Feedback**
- Descriptive enough to guide refinement
- Root cause analysis helps Opus understand issue
- Recommended fixes are concrete

✅ **Iterative Improvement**
- Each iteration addresses a class of errors
- Prompt becomes more robust over time
- Eventually reaches 100% through generalization

✅ **Maintains Validation Set Integrity**
- Test data truly measures generalization
- No information leakage of specific examples
- Scientifically defensible approach

### Potential Challenges

⚠️ **May Take More Iterations**
- Generic feedback is less direct than specific examples
- Opus may need multiple attempts to nail down edge cases
- Trade-off: More iterations for better generalization

⚠️ **Requires Smart Feedback Analysis**
- Need good error categorization system
- Pattern detection must be accurate
- Feedback analyzer is critical component

⚠️ **No Guarantee of Convergence**
- Theoretically possible that generic feedback insufficient
- In practice: With good descriptive feedback, should converge
- Fallback: Can increase feedback specificity if stuck after many iterations

---

## Implementation Details

### Feedback Analyzer Enhancement

**For Training Set:**
```python
def analyze_training_failures(failures):
    """Full details for training failures"""
    return {
        "case_id": case.id,
        "input": case.input,
        "expected": case.expected_output,
        "actual": case.actual_output,
        "diff": generate_diff(case.expected, case.actual),
        "error_category": categorize_error(case),
        "analysis": analyze_root_cause(case)
    }
```

**For Validation Set:**
```python
def analyze_validation_failures(failures):
    """Descriptive patterns without specific examples"""

    # Group failures by error category
    categorized = group_by_error_type(failures)

    feedback = []
    for category, cases in categorized.items():
        # Analyze pattern across cases WITHOUT revealing specifics
        pattern = extract_pattern_description(cases)

        feedback.append({
            "error_type": category,
            "count": len(cases),
            "pattern_observed": pattern.description,
            "example_pattern": pattern.generic_example,  # NOT actual input
            "root_cause": pattern.root_cause_hypothesis,
            "recommended_fix": pattern.suggested_guidance
        })

    return feedback
```

### Pattern Extraction Logic

```python
def extract_pattern_description(failed_cases):
    """Extracts common patterns without revealing specific inputs"""

    # Analyze error types
    error_types = [categorize_error(case) for case in failed_cases]

    # Find common characteristics (WITHOUT showing actual inputs)
    if all_are_format_errors(failed_cases):
        return Pattern(
            description="Output included additional text beyond required format",
            generic_example="Instead of outputting just 'X', output was 'X - explanation'",
            root_cause="Format specification not explicit enough",
            suggested_guidance="Add explicit constraint: Output ONLY the required value"
        )

    elif all_are_boundary_cases(failed_cases):
        return Pattern(
            description="Ambiguous cases near category boundaries misclassified",
            generic_example="Inputs with mixed signals chose wrong dominant category",
            root_cause="Boundary decision criteria unclear in instructions",
            suggested_guidance="Clarify how to resolve ambiguous cases"
        )

    # ... more pattern detection logic
```

---

## Success Criteria (Updated)

### Training Set
- **Target:** 100% pass rate
- **Feedback:** Full details (inputs, outputs, diffs)
- **Iterations:** Typically 5-10

### Validation Set
- **Target:** 100% pass rate
- **Feedback:** Descriptive patterns (no specific examples)
- **Iterations:** Typically 3-7 additional iterations
- **Convergence:** Continue until 100% achieved

### Total System
- **Success:** 100% on training + 100% on validation
- **Method:** Training optimization → Validation optimization → Held-out test evaluation
- **Quality:** Generalized prompt, not overfitted to specific examples

---

## Configuration Update

```yaml
optimizer:
  model: claude-opus-4.5
  max_iterations: 30  # Hard stop to prevent unbounded cost
  plateau_threshold: 7  # Stop if no improvement for N iterations

feedback:
  training_detail_level: full  # Show actual inputs/outputs
  test_detail_level: descriptive  # Pattern-based, no examples

  test_feedback_specificity:
    include_error_categories: true
    include_pattern_descriptions: true
    include_generic_examples: true
    include_root_cause_analysis: true
    include_recommended_fixes: true
    include_actual_validation_inputs: false  # ❌ Never
    include_actual_validation_outputs: false  # ❌ Never

stopping_criteria:
  # Success criteria (ideal outcome)
  training_pass_rate: 1.0  # Target: 100%
  test_pass_rate: 1.0      # Target: 100%

  # Cost controls (prevent unbounded spending)
  max_total_iterations: 30
  max_optimizer_tokens: 1000000  # ~$15 at Opus prices
  max_test_iterations: 15  # Don't spend more than 15 iterations on validation refinement

  # Plateau detection (stop if stuck)
  plateau_iterations: 7  # No improvement for N iterations
  minimum_improvement: 0.05  # Must improve by 5% to count as progress

  # What to do if limits hit without 100%
  on_incomplete_convergence: "return_best"  # Options: "return_best", "raise_error", "prompt_user"
```

---

## Cost Controls & Stopping Criteria

### The Reality: 100% May Not Always Be Achievable Within Budget

**Ideal:** Reach 100% on both training and validation sets

**Reality:** Must balance quality with cost

### Multi-Layer Stopping Criteria

#### 1. Success (Desired Outcome)
```
IF training_pass_rate == 100% AND test_pass_rate == 100%:
    STOP → Return optimized prompt ✅
```

#### 2. Iteration Limit (Hard Stop)
```
IF iteration_count >= max_total_iterations (30):
    STOP → Return best prompt so far ⚠️
    Report: "Reached iteration limit. Best result: Train 100%, Validation 95%"
```

**Rationale:** Prevents infinite loops and unbounded cost

#### 3. Token Budget (Cost Control)
```
IF total_optimizer_tokens >= max_optimizer_tokens (1M):
    STOP → Return best prompt so far ⚠️
    Report: "Reached token budget ($15). Best result: Train 100%, Validation 92%"
```

**Rationale:** Explicit cost cap for budget control

#### 4. Test Iteration Limit (Phase-Specific)
```
IF test_phase_iterations >= max_test_iterations (15):
    STOP → Return best prompt so far ⚠️
    Report: "Validation refinement limit reached. Train 100%, Validation 90%"
```

**Rationale:** If validation set isn't improving after 15 iterations of descriptive feedback, either:
- Validation set has fundamentally different distribution than training
- Descriptive feedback isn't specific enough
- Need more/better training data

#### 5. Plateau Detection (No Improvement)
```
IF no improvement for plateau_iterations (7) consecutive iterations:
    STOP → Return best prompt so far ⚠️
    Report: "Optimization plateaued. Train 100%, Validation 88%"
```

**Rationale:** If stuck at same performance for 7 iterations, unlikely to improve further

**Improvement Definition:**
```python
def is_improvement(current_score, previous_score, min_improvement=0.05):
    """Must improve by at least 5% to count as progress"""
    return (current_score - previous_score) >= min_improvement
```

### Decision Tree for Stopping

```
┌─────────────────────────────────────┐
│ Start Iteration N                   │
└──────────┬──────────────────────────┘
           │
           ▼
    ┌──────────────┐
    │ Train + Test │
    │  Both 100%?  │
    └──┬────────┬──┘
       │ Yes    │ No
       │        │
       ▼        ▼
    SUCCESS   ┌──────────────────┐
              │ Check Limits:    │
              │ 1. Iterations≥30?│
              │ 2. Tokens≥1M?    │
              │ 3. Test iters≥15?│
              │ 4. Plateau (7)?  │
              └──┬────────┬──────┘
                 │ Yes    │ No
                 │        │
                 ▼        ▼
           STOP (BEST)  CONTINUE
           ⚠️ Incomplete
```

### When Stopping Without 100%: What to Return

**Option 1: Return Best Prompt (Recommended)**
```python
result = {
    "status": "incomplete_convergence",
    "reason": "max_iterations_reached",  # or "plateau_detected", "budget_exceeded"
    "best_prompt": best_prompt_so_far,
    "metrics": {
        "training": {"pass_rate": 1.0, "cases": "20/20"},
        "validation": {"pass_rate": 0.95, "cases": "19/20"},
        "total_iterations": 30,
        "optimizer_tokens": 856432,
        "estimated_cost": "$12.85"
    },
    "recommendation": "Consider adding more training examples covering validation failure patterns"
}
```

**Option 2: Raise Error**
```python
raise OptimizationIncompleteError(
    "Failed to reach 100% on validation set within 30 iterations. "
    "Best result: Train 100%, Test 95%. "
    "Consider: (1) Increase max_iterations, (2) Add more training data, "
    "(3) Increase validation feedback specificity"
)
```

**Option 3: Prompt User for Decision**
```
⚠️ Optimization has not reached 100% on validation set.

Current Status:
- Training: 20/20 (100%)
- Validation: 19/20 (95%)
- Iterations used: 30/30
- Cost so far: $12.85

Options:
1. Accept current prompt (95% validation accuracy)
2. Continue optimization for 10 more iterations (+$4-5)
3. Show validation failure details to guarantee convergence
4. Add more training data and restart

Your choice [1-4]:
```

### Recommended Defaults

```yaml
stopping_criteria:
  # Targets
  training_pass_rate: 1.0
  test_pass_rate: 1.0

  # Hard limits (prevent runaway costs)
  max_total_iterations: 30         # ~$15-20 cost
  max_optimizer_tokens: 1000000    # ~$15 at Opus pricing
  max_test_iterations: 15          # Half of total for validation phase

  # Convergence detection
  plateau_iterations: 7
  minimum_improvement: 0.05        # 5% improvement threshold

  # Behavior on incomplete
  on_incomplete_convergence: "return_best"
  warn_user: true
  include_diagnostics: true
```

### Cost Estimation

**Per Iteration Costs (Approximate):**

Training Phase (full feedback):
- Optimizer call (Opus): ~15-30K tokens = $0.45-$0.90
- Test runner calls: (depends on target model)
- Total per iteration: ~$0.50-$1.00

Validation Phase (descriptive feedback):
- Optimizer call (Opus): ~10-20K tokens = $0.30-$0.60
- Test runner calls: (depends on target model)
- Total per iteration: ~$0.35-$0.75

**Total Budget Scenarios:**

Conservative (max 30 iterations):
- Training phase (10 iterations): $5-10
- Validation phase (15 iterations): $5-11
- Total: $10-21

Aggressive (unlimited with good convergence):
- Training phase (5 iterations): $2.50-5
- Validation phase (5 iterations): $1.75-3.75
- Total: $4.25-8.75

**Recommendation:** Set max_total_iterations to 30 for ~$20 maximum spend

### Diagnostic Information on Incomplete Convergence

When stopping without 100%, provide actionable diagnostics:

```python
diagnostics = {
    "remaining_failures": {
        "validation": [
            {
                "error_type": "Sarcasm Detection",
                "count": 1,
                "pattern": "Positive words in negative contexts",
                "attempts_to_fix": 8,  # We tried 8 times with descriptive feedback
                "recommendation": "This pattern resisted generic feedback. Consider: "
                                  "(1) Add similar examples to training set, or "
                                  "(2) Manually craft sarcasm guidance"
            }
        ]
    },
    "iteration_history": [
        {"iter": 1, "train": 0.60, "validation": None},
        {"iter": 5, "train": 1.00, "validation": None},
        {"iter": 6, "train": 1.00, "validation": 0.80},
        {"iter": 10, "train": 1.00, "validation": 0.90},
        {"iter": 20, "train": 1.00, "validation": 0.95},
        {"iter": 30, "train": 1.00, "validation": 0.95},  # Plateaued at 95%
    ],
    "plateau_detected_at": 20,  # No improvement after iteration 20
    "suggestions": [
        "Validation set performance plateaued at 95% after iteration 20",
        "Remaining failure: Sarcasm detection (1 case) resisted 8 refinement attempts",
        "Recommended: Add sarcasm examples to training set",
        "Alternative: Increase test_feedback_specificity or show actual validation case"
    ]
}
```

---

## Example: Full Optimization Run

### Iteration 1-5: Training Phase

```
Iteration 1: 12/20 training (60%)
→ Full feedback on 8 failures
→ Opus adds format specification

Iteration 2: 17/20 training (85%)
→ Full feedback on 3 failures
→ Opus adds mixed sentiment guidance

Iteration 3: 20/20 training (100%) ✅
→ Training complete, proceed to validation
```

### Iteration 6: First Validation

```
Validation Results: 8/10 (80%)

Descriptive Feedback:
- Sarcasm detection issues (2 cases)
  Pattern: Positive words in negative contexts
  Fix: Add contextual interpretation guide
```

### Iteration 7: Refined After Validation Feedback

```
Opus adds sarcasm handling to prompt

Validation Results: 9/10 (90%)

Descriptive Feedback:
- Neutral boundary issue (1 case)
  Pattern: Factual statement misclassified
  Fix: Clarify neutral = absence of sentiment
```

### Iteration 8: Second Refinement

```
Opus clarifies neutral definition

Validation Results: 10/10 (100%) ✅

SUCCESS:
- Training: 20/20 (100%)
- Validation: 10/10 (100%)
- Total iterations: 8
- Final prompt: Generalized, not overfit
```

### Phase 3: Held-out Test Evaluation

```
Held-out Results: 9/10 (90%)
Report only (no feedback provided during optimization)
```

---

## Why This Achieves 100% Without Overfitting

1. **Training Phase:** Optimizer learns from detailed examples
2. **Validation Phase:** Optimizer refines based on error *patterns*, not specific cases
3. **Generalization:** Fixes address classes of errors, not individual examples
4. **Validation:** Validation set truly measures generalization since specifics never revealed
5. **Convergence:** Descriptive feedback is actionable enough to guide refinement to 100%

This approach balances the competing goals of:
- ✅ Reaching 100% on validation set (hard requirement)
- ✅ Avoiding overfitting (don't show validation cases)
- ✅ Maintaining scientific rigor (validation set integrity)
- ✅ Actionable feedback (descriptive enough to fix issues)

---

## Recommendation

**This strategy should be our default approach:**

1. **Phase 1:** Optimize on training with full feedback → 100%
2. **Phase 2:** Validate on validation with descriptive feedback → iterate until 100%
3. **Success:** Both sets at 100%, prompt generalized not memorized
4. **Iterations:** May take 20-30 total, but worth it for quality

This is the right balance for your requirements.
