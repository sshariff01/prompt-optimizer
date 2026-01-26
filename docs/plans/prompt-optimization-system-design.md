# Prompt Optimization System Design

**Date:** 2026-01-25
**Status:** Initial Design - Exploring Approaches

## Problem Statement

Build a system that automatically generates and optimizes prompts using eval-based feedback loops. The system should:
- Work for any task where users can define their own evals
- Optimize prompts for multiple closely related tasks
- Use reinforcement learning principles to iteratively improve
- Pass all user-provided evals or provide clear feedback on progress

## Requirements Summary

**Task Scope:**
- General-purpose optimizer
- Users define their own evaluation criteria
- Target: Multiple closely related tasks

**Evaluation Format:**
- Input/output pairs: Simple test cases with expected outputs
- Programmatic assertions: Code-based tests checking output properties

**Optimization Focus:**
- Prompt text only (not model parameters, few-shot examples, or system instructions)

**Priority:**
- Quality over speed - willing to use expensive models and more iterations

**Feedback Mechanism:**
- Pass/fail scores (e.g., "7/10 passed")
- Diff between expected and actual outputs
- Error analysis with categorized failure types
- Comprehensive feedback to maximize learning signal

## Architectural Approaches

### Approach 1: Iterative Meta-Prompt Refinement (Recommended)

**How it works:**
A powerful model (Opus 4.5) acts as the "optimizer" that iteratively refines prompts. Each iteration:

1. Optimizer generates/refines a prompt
2. Test runner executes the prompt against all evals using a separate model (configurable)
3. Comprehensive feedback (scores, diffs, error analysis) goes back to optimizer
4. Optimizer analyzes failures and generates an improved prompt
5. Repeat until all evals pass or max iterations reached

**Pros:**
- Opus 4.5 has excellent meta-reasoning abilities for prompt engineering
- Simple architecture with clear separation of concerns
- Comprehensive feedback naturally fits into the optimization loop
- Quality-focused approach that leverages best available models
- Most reliable convergence

**Cons:**
- More expensive (Opus calls for optimization)
- Slower than parallel approaches
- Sequential process (can't parallelize optimization itself)

**Why Recommended:**
Since quality is the priority, the cost of Opus is justified. The comprehensive feedback loop works naturally with iterative refinement, and this approach has the highest reliability for convergence.

---

### Approach 2: Evolutionary Prompt Optimization

**How it works:**
Maintain a population of candidate prompts:

1. Generate initial population (5-10 prompts) using Sonnet 4.5
2. Evaluate all candidates in parallel against evals
3. Select best performers, create variations (mutations/crossovers)
4. Repeat for multiple generations
5. Return best prompt from final generation

**Pros:**
- Explores solution space more broadly
- Can parallelize eval runs across candidates
- May discover unexpected approaches
- Less likely to get stuck in local optima

**Cons:**
- More complex implementation
- May not converge as reliably
- Higher token costs (evaluating multiple candidates)
- Feedback integration is less direct
- Requires careful tuning of population size, mutation rates, etc.

---

### Approach 3: Hybrid - Evolutionary Start, Iterative Refinement

**How it works:**
Combines both approaches:

1. **Phase 1 (Exploration):** Generate 3-5 diverse initial prompts using Sonnet
2. Evaluate all candidates and pick the best
3. **Phase 2 (Exploitation):** Switch to iterative refinement using Opus on the winner
4. Continue until convergence

**Pros:**
- Balances exploration and exploitation
- Starts from a better initial position
- Less risk of local optima than pure iterative
- Still gets refinement quality of Opus

**Cons:**
- More complex orchestration
- Longer overall process
- Requires careful phase transition logic
- May not be worth the added complexity

## Deep Dive: Approach 1 vs Approach 2 Tradeoff Analysis

### Will They Converge to the Same Solution?

**No - they will likely converge to different solutions**, for several fundamental reasons:

1. **Multiple Local Optima**: The prompt optimization space contains many local optima (different prompts that work equally well). Approach 1 converges to the nearest local optimum from its starting point, while Approach 2 explores multiple regions simultaneously and may find different local optima.

2. **Path Dependence**: In Approach 1, iteration N depends heavily on iteration N-1 - it follows a specific gradient. Approach 2 lacks this path dependence since it evaluates diverse candidates in parallel with independent mutation pathways.

3. **Non-deterministic Generation**: Both approaches use LLMs with temperature/randomness, so even running the same approach twice would yield different results.

4. **Fundamentally Different Search Strategies**: Approach 1 is "smart hill-climbing" with rich gradient information, while Approach 2 is population-based evolutionary search - fundamentally different optimization algorithms.

### Why Approach 2 Is NOT Objectively Better (Despite More Exploration)

A key insight from optimization theory: **more exploration ≠ better outcomes**. Here's why:

#### 1. Exploration-Exploitation Trade-off
- **Approach 2**: Spreads computational resources across many mediocre solutions
- **Approach 1**: Focuses all resources on deeply refining one promising solution
- With a high-quality reasoner (Opus 4.5), deep exploitation is often more efficient than broad exploration

#### 2. Quality of Search Direction
- **Approach 1** receives **rich gradient information** from comprehensive feedback
- Opus can analyze failure patterns and make intelligent, targeted refinements
- This "smart" directed search is more efficient than "random" evolutionary exploration
- Each iteration provides high signal-to-noise ratio for the next refinement
- **Approach 2** relies on random mutations and crossover, which are less intelligent

#### 3. Sample Efficiency
- **Approach 2**: Evaluates 5-10 candidates × N generations = 50-100+ evaluations
- **Approach 1**: Typically 5-10 total iterations to convergence
- Each evaluation costs tokens, time, and API calls
- Approach 1 extracts more "learning" per evaluation through detailed feedback analysis

#### 4. Convergence Speed
- Evolutionary algorithms can be slow to converge - populations can stagnate
- Iterative refinement with strong feedback often converges faster to "good enough"
- If both eventually pass all evals, the faster approach is superior

#### 5. Success Criteria: Binary Not Optimal
- Success criteria: **"pass all evals"** (binary), not "find the globally optimal prompt"
- There are likely **many prompts** that achieve 100% eval pass rate
- Finding **a solution quickly** matters more than finding **the theoretically best solution**
- Approach 1 might reach 100% in 5 iterations; Approach 2 might need 20 generations

#### 6. The "No Free Lunch" Theorem
From optimization theory, there's no universally superior algorithm. The best approach depends on problem structure:
- For problems with **informative gradients** (like detailed eval feedback) → gradient-based methods win
- For problems with **deceptive or noisy fitness landscapes** → evolutionary methods win
- Our problem has rich, informative feedback, favoring Approach 1

#### 7. Diminishing Returns from Exploration
- Once in a "basin of attraction" around good solutions, deep refinement matters more than exploration
- Evolutionary exploration is most valuable when you don't know where good regions are
- With comprehensive feedback guiding you, additional exploration becomes less critical

### When Approach 2 WOULD Be Superior

To be fair, evolutionary is better when:
- **Starting point is poor**: Less sensitive to initial conditions
- **Feedback is noisy**: Population averaging provides robustness
- **Multiple diverse solutions needed**: Naturally generates variety
- **Creative/unexpected solutions valued**: Mutations discover non-obvious approaches
- **Search space is deceptive**: Gradients point toward poor local minima
- **Parallelization is critical**: Can utilize many compute resources simultaneously

### Cost-Benefit Analysis

#### Approach 1: Iterative Refinement
- **Cost**: ~5-10 Opus calls for optimization + 5-10 eval runs
- **Benefit**: Fast convergence, high-quality reasoning, simple implementation
- **Time to Solution**: 5-10 iterations (serial)

#### Approach 2: Evolutionary
- **Cost**: ~50-100+ Sonnet calls + 50-100+ eval runs
- **Benefit**: Broader exploration, potentially discovers creative solutions
- **Time to Solution**: 10-20 generations with 5-10 candidates each (parallelizable)

### Real-World Analogy

Think of finding a restaurant:

**Approach 1**: Ask a food critic (Opus) to analyze what's wrong with each restaurant you try, then recommend a better one nearby. Fast convergence to a great restaurant in your area.

**Approach 2**: Send 10 people to random restaurants across the city, pick the best 3, spawn 10 new restaurant visits based on "mutations" of what worked. Eventually finds great restaurants, but takes longer and costs more.

## Enhancement: Optimization Memory (Context Preservation)

### Problem: Memoryless Iterations

In the initial implementation of Approach 1, each iteration was effectively independent:
- ✅ Optimizer sees current prompt and current failures
- ❌ Optimizer does NOT see previous prompts that were tried
- ❌ Optimizer does NOT see what changes were made and why
- ❌ Optimizer does NOT see which approaches worked vs failed
- ❌ Optimizer does NOT see historical failures that were fixed

**Result:** The meta-optimizer is essentially memoryless between iterations, like having amnesia. It might:
- Try the same failed approaches repeatedly
- Undo previous successful fixes
- Not learn from patterns across iterations
- Miss compounding insights

### Solution: Hybrid Memory Architecture

We implemented a **two-tier memory system** that provides both short-term and long-term context:

#### 1. Accumulated Lessons (Long-term Memory)
Maintains a running list of insights extracted from each iteration:
- `✓ Added explicit format examples → improved performance`
- `✗ Over-specified edge cases → caused regression`

Provides general principles learned throughout optimization.

#### 2. Recent Iteration History (Short-term Memory)
Keeps detailed summaries of the last 3 iterations:

```
Iteration N: ✓ ACCEPTED
  Change Made: Addressed format_violation, boundary_confusion errors
  Target Issues: 15 format_violation errors, 5 boundary_confusion errors
  Result: Training 80% → 90%
          Fixed 10 failures, improved by 10%
```

Provides concrete trajectory information showing what was recently attempted.

### Context Integration

On each refinement, the meta-optimizer receives:

```
Optimization Context:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accumulated Lessons:
• ✓ Explicit format examples improve compliance
• ✗ Over-specifying edge cases causes brittleness

Recent Iteration History:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Last 3 iterations with details...]

Current Prompt:
[current prompt]

Training Results: [failures...]

Your task: Build on what worked, avoid what failed, address current issues.
```

### Benefits

The optimization memory enables:
- **Learning from mistakes** - Won't repeat rejected approaches
- **Building on successes** - Knows what patterns worked before
- **Understanding trajectory** - Sees the optimization journey
- **Making informed decisions** - Has context to avoid breaking what works
- **Compounding insights** - Accumulates wisdom across iterations

### Implementation

**Components:**
- `OptimizationMemory`: Maintains iteration history and lessons
- `IterationSummary`: Detailed record of each iteration (changes, issues, outcomes)
- Helper methods to extract issues, describe changes, and generate outcomes

**Token Cost:**
- ~500-1000 tokens per iteration for context
- Offset by improved convergence (fewer total iterations needed)
- Net positive ROI due to better decision-making

## Recommended Approach

**We recommend Approach 1: Iterative Meta-Prompt Refinement with Optimization Memory**

### Rationale:

1. **Aligns with Quality Priority**: You specified quality over speed - Opus 4.5 provides the highest quality meta-reasoning for prompt engineering

2. **Sample Efficiency**: Given comprehensive feedback (scores, diffs, error analysis), directed search extracts maximum learning from each evaluation

3. **Cost-Effective for Quality**: While Opus is expensive per call, total cost is lower due to fewer iterations needed (5-10 vs 50-100+)

4. **Implementation Simplicity**: Simpler architecture reduces bugs, easier to debug, faster to deploy

5. **Reliable Convergence**: Binary success criteria ("pass all evals") suits gradient-based optimization well

6. **Informative Feedback Available**: Your comprehensive feedback mechanism provides rich gradients that Approach 1 can exploit effectively

7. **Proven Track Record**: Similar architectures (DSPy, TextGrad) have shown success with iterative prompt optimization

### Success Metrics:
- **Primary**: % of evals passed (target: 100%)
- **Secondary**: Number of iterations to convergence
- **Tertiary**: Total cost (API tokens) and wall-clock time

### Fallback Strategy:
If Approach 1 consistently fails to converge after 15-20 iterations across multiple runs, consider implementing Approach 3 (Hybrid) to add initial exploration phase.

## Next Steps

1. Validate architectural direction with stakeholder
2. Proceed with Approach 1 detailed component design:
   - Optimizer component (meta-prompt engineering with Opus 4.5)
   - Test runner component (eval execution)
   - Feedback analyzer (error categorization and diff generation)
   - Orchestrator (iteration control and convergence detection)
3. Design the meta-prompt template for the optimizer
4. Define stopping criteria (max iterations, convergence detection)
5. Implement evaluation harness for both input/output pairs and programmatic assertions
6. Build feedback formatting system for rich gradient signals
