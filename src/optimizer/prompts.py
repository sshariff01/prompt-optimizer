"""Meta-prompt templates for the optimizer."""

from jinja2 import Template

# Initial prompt generation template
INITIAL_PROMPT_GENERATION = Template(
    """You are a prompt engineering expert. Your task is to generate a zero-shot instruction prompt for the following task:

{{ task_description }}

Here are example input/output pairs showing the desired behavior:
{% for example in training_examples %}
Input: {{ example.input }}
Expected Output: {{ example.expected_output }}
{% endfor %}

Generate a clear, concise zero-shot prompt that will perform this task correctly. The prompt should include:
- Clear instructions
- Format specifications
- Guidance for edge cases

IMPORTANT: The input will be provided AFTER your prompt in the format "Input: <user_input>".
Do NOT include input placeholders like {input} or {{input}} in your prompt.
Your prompt should end with instructions, and the actual input will be appended automatically.

Output ONLY the prompt text, no explanations."""
)

# Refinement prompt template (for training set feedback)
REFINEMENT_PROMPT_TRAINING = Template(
    """You are refining a prompt that is not performing perfectly on the training set.

{% if optimization_context %}
{{ optimization_context }}

{% endif %}
Current Prompt:
{{ current_prompt }}

Training Set Evaluation Results:
Pass Rate: {{ pass_rate }}% ({{ passed }}/{{ total }} passed)

Detailed Failures:
{% for failure in failures %}
---
Case {{ loop.index }}:
  Input: {{ failure.case.input }}
  Expected Output: {{ failure.case.expected_output }}
  Actual Output: {{ failure.actual_output }}

  Diff: {{ failure.diff }}

  Error Category: {{ failure.error_category }}
  Analysis: {{ failure.analysis }}
{% endfor %}

Your task: Analyze these failures and generate an improved version of the prompt.
Focus on addressing the specific error patterns shown in the feedback.

IMPORTANT: The input will be provided AFTER your prompt in the format "Input: <user_input>".
Do NOT include input placeholders like {input} or {{input}} in your prompt.

Output ONLY the refined prompt text, no explanations."""
)

# Refinement prompt template (for test set feedback - descriptive only)
REFINEMENT_PROMPT_TEST = Template(
    """You are refining a prompt based on test set validation results.

{% if optimization_context %}
{{ optimization_context }}

{% endif %}
Current Prompt:
{{ current_prompt }}

Note: The training set passed 100%. The test set reveals generalization issues.

Test Set Evaluation Results:
Pass Rate: {{ pass_rate }}% ({{ passed }}/{{ total }} passed)

Error Patterns Observed (NO specific test cases shown):
{% for pattern in error_patterns %}
---
Error Type {{ loop.index }}: {{ pattern.error_type }} ({{ pattern.count }} failures)

  Pattern Observed: {{ pattern.pattern_observed }}

  Example Pattern: {{ pattern.example_pattern }}

  Root Cause: {{ pattern.root_cause }}

  Recommended Fix: {{ pattern.recommended_fix }}
{% endfor %}

Your task: Analyze these error patterns and generate an improved version of the prompt.
Focus on generalizing the prompt to handle these patterns, NOT on memorizing specific cases.
The goal is to make the prompt more robust without overfitting to the test set.

IMPORTANT: The input will be provided AFTER your prompt in the format "Input: <user_input>".
Do NOT include input placeholders like {input} or {{input}} in your prompt.

Output ONLY the refined prompt text, no explanations."""
)

# Combined refinement prompt template (test patterns + training constraints)
REFINEMENT_PROMPT_COMBINED = Template(
    """You are refining a prompt with CONSTRAINTS. Your previous refinement caused training set regression.

{% if optimization_context %}
{{ optimization_context }}

{% endif %}
Current Prompt:
{{ current_prompt }}

SITUATION: Your last refinement attempt tried to fix test set issues but broke the training set.

Test Set Issues (what you were trying to fix):
Pass Rate: {{ test_pass_rate }}% ({{ test_passed }}/{{ test_total }} passed)

Error Patterns to Address:
{% for pattern in error_patterns %}
---
Error Type {{ loop.index }}: {{ pattern.error_type }} ({{ pattern.count }} failures)

  Pattern Observed: {{ pattern.pattern_observed }}

  Example Pattern: {{ pattern.example_pattern }}

  Root Cause: {{ pattern.root_cause }}

  Recommended Fix: {{ pattern.recommended_fix }}
{% endfor %}

Training Set Constraints (cases that broke in your last attempt):
Pass Rate: {{ training_pass_rate }}% ({{ training_passed }}/{{ training_total }} passed)

Training Failures You Must NOT Break:
{% for failure in training_failures %}
---
Case {{ loop.index }}:
  Input: {{ failure.case.input }}
  Expected Output: {{ failure.case.expected_output }}
  What Your Last Refinement Produced: {{ failure.actual_output }}

  Diff: {{ failure.diff }}

  Error Category: {{ failure.error_category }}
  Analysis: {{ failure.analysis }}
{% endfor %}

Your task: Generate an improved prompt that fixes the TEST patterns WITHOUT breaking these TRAINING cases.
This requires careful balancing - you must improve test generalization while maintaining training performance.

Strategies to consider:
- Add nuanced rules that handle test patterns while preserving training behavior
- Use conditional logic or examples that cover both training and test scenarios
- Avoid over-corrections that sacrifice training accuracy for test improvements

IMPORTANT: The input will be provided AFTER your prompt in the format "Input: <user_input>".
Do NOT include input placeholders like {input} or {{input}} in your prompt.

Output ONLY the refined prompt text, no explanations."""
)


def render_initial_prompt(task_description: str, training_examples: list) -> str:
    """Render the initial prompt generation template.

    Args:
        task_description: Description of the task
        training_examples: List of EvalCase objects

    Returns:
        Rendered prompt for Opus
    """
    return INITIAL_PROMPT_GENERATION.render(
        task_description=task_description, training_examples=training_examples
    )


def render_training_refinement(
    current_prompt: str,
    pass_rate: float,
    passed: int,
    total: int,
    failures: list,
    optimization_context: str = "",
) -> str:
    """Render the training set refinement prompt.

    Args:
        current_prompt: The current prompt being refined
        pass_rate: Pass rate as percentage
        passed: Number of cases passed
        total: Total number of cases
        failures: List of FailureAnalysis objects
        optimization_context: Context from previous iterations

    Returns:
        Rendered prompt for Opus
    """
    return REFINEMENT_PROMPT_TRAINING.render(
        current_prompt=current_prompt,
        pass_rate=pass_rate * 100,
        passed=passed,
        total=total,
        failures=failures,
        optimization_context=optimization_context,
    )


def render_test_refinement(
    current_prompt: str,
    pass_rate: float,
    passed: int,
    total: int,
    error_patterns: list,
    optimization_context: str = "",
) -> str:
    """Render the test set refinement prompt.

    Args:
        current_prompt: The current prompt being refined
        pass_rate: Pass rate as percentage
        passed: Number of cases passed
        total: Total number of cases
        error_patterns: List of ErrorPattern objects
        optimization_context: Context from previous iterations

    Returns:
        Rendered prompt for Opus
    """
    return REFINEMENT_PROMPT_TEST.render(
        current_prompt=current_prompt,
        pass_rate=pass_rate * 100,
        passed=passed,
        total=total,
        error_patterns=error_patterns,
        optimization_context=optimization_context,
    )


def render_combined_refinement(
    current_prompt: str,
    test_pass_rate: float,
    test_passed: int,
    test_total: int,
    error_patterns: list,
    training_pass_rate: float,
    training_passed: int,
    training_total: int,
    training_failures: list,
    optimization_context: str = "",
) -> str:
    """Render the combined refinement prompt (test patterns + training constraints).

    Args:
        current_prompt: The current prompt being refined
        test_pass_rate: Test pass rate as percentage
        test_passed: Number of test cases passed
        test_total: Total number of test cases
        error_patterns: List of ErrorPattern objects from test set
        training_pass_rate: Training pass rate after regression
        training_passed: Number of training cases passed
        training_total: Total number of training cases
        training_failures: List of FailureAnalysis objects from training regression
        optimization_context: Context from previous iterations

    Returns:
        Rendered prompt for Opus
    """
    return REFINEMENT_PROMPT_COMBINED.render(
        current_prompt=current_prompt,
        test_pass_rate=test_pass_rate * 100,
        test_passed=test_passed,
        test_total=test_total,
        error_patterns=error_patterns,
        training_pass_rate=training_pass_rate * 100,
        training_passed=training_passed,
        training_total=training_total,
        training_failures=training_failures,
        optimization_context=optimization_context,
    )
