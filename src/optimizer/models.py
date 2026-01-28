"""Pydantic models for configuration, data, and results."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class TargetModelConfig(BaseModel):
    """Configuration for the target model being optimized for."""

    provider: str = Field(..., description="Provider name (e.g., 'anthropic', 'openai')")
    model: str = Field(..., description="Model name (e.g., 'claude-sonnet-4.5')")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, gt=0)


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer (meta-prompt engineer)."""

    model: str = Field(
        default="claude-opus-4.5",
        description="Model to use for optimization (default: claude-opus-4.5)",
    )
    max_iterations: int = Field(default=30, gt=0, description="Maximum optimization iterations")
    max_optimizer_tokens: int = Field(
        default=1_000_000, gt=0, description="Maximum tokens for optimizer calls"
    )
    max_test_iterations: int = Field(
        default=15, gt=0, description="Maximum iterations for test set refinement"
    )
    plateau_threshold: int = Field(
        default=7, gt=0, description="Stop if no improvement for N iterations"
    )
    max_workers: int = Field(
        default=50, gt=0, description="Maximum parallel API calls for evaluation (default: 50)"
    )
    candidates_per_iteration: int = Field(
        default=3, gt=0, le=10, description="Number of candidate prompts to generate per iteration (default: 3)"
    )


class DataConfig(BaseModel):
    """Configuration for training and test data."""

    training_set: Path = Field(..., description="Path to training set JSONL file")
    test_set: Path = Field(..., description="Path to test set JSONL file")


class SchemaConfig(BaseModel):
    """Configuration for strict output label schemas."""

    fields: list[str] = Field(
        ..., description="Ordered list of required output fields (e.g., ['LABEL'])"
    )
    enums: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Allowed enum values per field (case-insensitive)",
    )
    patterns: dict[str, str] = Field(
        default_factory=dict,
        description="Regex patterns per field (case-insensitive match)",
    )


class FeedbackDetailLevel(str, Enum):
    """Level of detail in feedback."""

    FULL = "full"  # Show all details (inputs, outputs, diffs)
    DESCRIPTIVE = "descriptive"  # Pattern-based without specific examples


class FeedbackConfig(BaseModel):
    """Configuration for feedback generation."""

    training_detail_level: FeedbackDetailLevel = FeedbackDetailLevel.FULL
    test_detail_level: FeedbackDetailLevel = FeedbackDetailLevel.DESCRIPTIVE


class IncompleteConvergenceStrategy(str, Enum):
    """What to do if optimization doesn't reach 100%."""

    RETURN_BEST = "return_best"  # Return best prompt found
    RAISE_ERROR = "raise_error"  # Raise an exception
    PROMPT_USER = "prompt_user"  # Ask user what to do


class StoppingCriteriaConfig(BaseModel):
    """Configuration for stopping criteria."""

    training_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    test_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    on_incomplete_convergence: IncompleteConvergenceStrategy = (
        IncompleteConvergenceStrategy.RETURN_BEST
    )


class OptimizationConfig(BaseSettings):
    """Main configuration for the optimization system."""

    task_description: str = Field(..., description="Description of the task to optimize for")
    target_model: TargetModelConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfig
    schema: SchemaConfig | None = Field(
        default=None, description="Optional strict output schema for evaluation"
    )
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    stopping_criteria: StoppingCriteriaConfig = Field(default_factory=StoppingCriteriaConfig)

    class Config:
        """Pydantic config."""

        env_prefix = "PROMPT_OPT_"


class EvalCase(BaseModel):
    """A single evaluation case (input/output pair)."""

    input: str = Field(..., description="Input to the prompt")
    expected_output: str = Field(..., description="Expected output from the prompt")


class EvalResult(BaseModel):
    """Result of evaluating a single case."""

    case: EvalCase
    actual_output: str
    passed: bool
    error: str | None = None


class ErrorCategory(str, Enum):
    """Categories of errors in eval results."""

    FORMAT_VIOLATION = "format_violation"  # Wrong output format
    BOUNDARY_CONFUSION = "boundary_confusion"  # Edge case mishandling
    LOGIC_ERROR = "logic_error"  # Incorrect reasoning
    MISSING_INFO = "missing_info"  # Output missing required information
    TOO_VERBOSE = "too_verbose"  # Output contains extra information
    OTHER = "other"  # Uncategorized error


class FailureAnalysis(BaseModel):
    """Detailed analysis of a failure."""

    case: EvalCase
    actual_output: str
    diff: str
    error_category: ErrorCategory
    analysis: str  # Human-readable explanation of the error


class DetailedFeedback(BaseModel):
    """Comprehensive feedback for training set optimization."""

    pass_rate: float
    passed: int
    total: int
    failures: list[FailureAnalysis]


class ErrorPattern(BaseModel):
    """A pattern of errors observed in test set (without specific examples)."""

    error_type: ErrorCategory
    count: int
    pattern_observed: str  # Generic description
    example_pattern: str  # Generic example (NOT actual test case)
    root_cause: str  # Hypothesis about why this is happening
    recommended_fix: str  # Specific guidance for fixing


class DescriptiveFeedback(BaseModel):
    """Pattern-based feedback for test set (avoids overfitting)."""

    pass_rate: float
    passed: int
    total: int
    error_patterns: list[ErrorPattern]


class CombinedFeedback(BaseModel):
    """Combined feedback when test refinement causes training regression.

    Includes both test patterns to fix and training constraints to maintain.
    """

    test_pass_rate: float
    test_passed: int
    test_total: int
    error_patterns: list[ErrorPattern]  # Test patterns to fix

    training_pass_rate: float
    training_passed: int
    training_total: int
    training_failures: list[FailureAnalysis]  # Training cases that broke


class IterationResult(BaseModel):
    """Result of a single optimization iteration."""

    iteration: int
    prompt: str
    training_pass_rate: float | None = None
    test_pass_rate: float | None = None
    feedback: DetailedFeedback | DescriptiveFeedback | CombinedFeedback | None = None
    optimizer_tokens_used: int = 0


class OptimizationStatus(str, Enum):
    """Status of the optimization run."""

    SUCCESS = "success"  # Reached target pass rates
    MAX_ITERATIONS = "max_iterations"  # Hit iteration limit
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"  # Hit token limit
    PLATEAU_DETECTED = "plateau_detected"  # No improvement
    ERROR = "error"  # Error occurred


class OptimizationResult(BaseModel):
    """Final result of an optimization run."""

    status: OptimizationStatus
    final_prompt: str
    iterations: list[IterationResult]
    training_pass_rate: float
    test_pass_rate: float
    total_optimizer_tokens: int
    message: str  # Human-readable status message
