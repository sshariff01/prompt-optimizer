"""Feedback analyzer for generating rich feedback from eval results."""

import difflib
from typing import List

from src.optimizer.models import (
    DescriptiveFeedback,
    DetailedFeedback,
    ErrorCategory,
    ErrorPattern,
    EvalResult,
    FailureAnalysis,
)


class FeedbackAnalyzer:
    """Analyzes evaluation results and generates structured feedback."""

    def analyze_training_results(self, results: list[EvalResult]) -> DetailedFeedback:
        """Analyze training set results with full details.

        Args:
            results: List of evaluation results

        Returns:
            Detailed feedback with full failure information
        """
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0.0

        # Get failures and analyze each one
        failures = [r for r in results if not r.passed]
        failure_analyses = []

        for failure in failures:
            analysis = self._analyze_failure(failure)
            failure_analyses.append(analysis)

        return DetailedFeedback(
            pass_rate=pass_rate,
            passed=passed,
            total=total,
            failures=failure_analyses,
        )

    def analyze_test_results(self, results: list[EvalResult]) -> DescriptiveFeedback:
        """Analyze test set results with descriptive patterns only (no specifics).

        This prevents overfitting by not revealing actual test cases.

        Args:
            results: List of evaluation results

        Returns:
            Descriptive feedback with generic patterns
        """
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0.0

        # Get failures and group by category
        failures = [r for r in results if not r.passed]

        # Group failures by error category
        categorized = {}
        for failure in failures:
            category = self._categorize_error(failure)
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(failure)

        # Generate error patterns for each category
        error_patterns = []
        for category, cases in categorized.items():
            pattern = self._extract_error_pattern(category, cases)
            error_patterns.append(pattern)

        return DescriptiveFeedback(
            pass_rate=pass_rate,
            passed=passed,
            total=total,
            error_patterns=error_patterns,
        )

    def _extract_error_pattern(
        self, category: ErrorCategory, cases: list[EvalResult]
    ) -> ErrorPattern:
        """Extract a generic error pattern from multiple failure cases.

        This generates descriptive feedback WITHOUT revealing specific test inputs/outputs.

        Args:
            category: The error category
            cases: List of failures in this category

        Returns:
            Error pattern with generic descriptions
        """
        # Pattern descriptions for each category
        patterns = {
            ErrorCategory.FORMAT_VIOLATION: {
                "pattern_observed": "Output format doesn't match the required structure",
                "example_pattern": (
                    "Instead of outputting just the required value, "
                    "output included additional structure or formatting"
                ),
                "root_cause": "Instructions not explicit enough about output format requirements",
                "recommended_fix": (
                    "Add explicit format constraint: "
                    "Output must contain ONLY the required value with no additional text"
                ),
            },
            ErrorCategory.TOO_VERBOSE: {
                "pattern_observed": "Output included additional explanatory text beyond the required output",
                "example_pattern": (
                    "Output contains phrases like 'because...', '- reasoning', "
                    "or other explanations alongside the required value"
                ),
                "root_cause": "Instructions don't emphasize that ONLY the answer should be provided",
                "recommended_fix": (
                    "Emphasize in instructions: Output ONLY the classification/answer "
                    "with no explanations, reasoning, or additional commentary"
                ),
            },
            ErrorCategory.MISSING_INFO: {
                "pattern_observed": "Output is incomplete or missing required information",
                "example_pattern": "Expected output components are absent or truncated",
                "root_cause": "Instructions unclear about all required output components",
                "recommended_fix": (
                    "Clarify what information must be included in every output. "
                    "List all required components explicitly."
                ),
            },
            ErrorCategory.BOUNDARY_CONFUSION: {
                "pattern_observed": "Ambiguous or edge case inputs were classified incorrectly",
                "example_pattern": (
                    "Cases with mixed signals, unclear boundaries between categories, "
                    "or inputs that could reasonably fit multiple classifications"
                ),
                "root_cause": "Instructions lack clear guidance for resolving ambiguous cases",
                "recommended_fix": (
                    "Add guidance for handling edge cases: "
                    "When inputs have mixed or conflicting signals, specify how to determine "
                    "the dominant category or how to break ties"
                ),
            },
            ErrorCategory.LOGIC_ERROR: {
                "pattern_observed": "Incorrect interpretation or reasoning about the input",
                "example_pattern": (
                    "Model's understanding of the task or input semantics "
                    "differs from intended interpretation"
                ),
                "root_cause": "Task description or examples don't cover this type of reasoning",
                "recommended_fix": (
                    "Clarify task semantics and provide more explicit reasoning guidelines. "
                    "Consider adding examples that demonstrate the correct interpretation."
                ),
            },
            ErrorCategory.OTHER: {
                "pattern_observed": "Unspecified error pattern",
                "example_pattern": "Error type doesn't fit standard categories",
                "root_cause": "Unable to determine root cause from error patterns",
                "recommended_fix": "Review instructions for clarity and completeness",
            },
        }

        pattern_info = patterns.get(category, patterns[ErrorCategory.OTHER])

        return ErrorPattern(
            error_type=category,
            count=len(cases),
            pattern_observed=pattern_info["pattern_observed"],
            example_pattern=pattern_info["example_pattern"],
            root_cause=pattern_info["root_cause"],
            recommended_fix=pattern_info["recommended_fix"],
        )

    def _analyze_failure(self, result: EvalResult) -> FailureAnalysis:
        """Analyze a single failure in detail.

        Args:
            result: The failed evaluation result

        Returns:
            Detailed failure analysis
        """
        # Generate diff
        diff = self._generate_diff(result.case.expected_output, result.actual_output)

        # Categorize error
        category = self._categorize_error(result)

        # Generate analysis
        analysis = self._generate_analysis(result, category)

        return FailureAnalysis(
            case=result.case,
            actual_output=result.actual_output,
            diff=diff,
            error_category=category,
            analysis=analysis,
        )

    def _generate_diff(self, expected: str, actual: str) -> str:
        """Generate a readable diff between expected and actual.

        Args:
            expected: Expected output
            actual: Actual output

        Returns:
            Formatted diff string
        """
        # Use difflib for character-level diff
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )

        diff_text = "".join(diff)

        if not diff_text:
            # If no line-level diff, show simple comparison
            return f'Expected: "{expected}"\nActual: "{actual}"'

        return diff_text

    def _categorize_error(self, result: EvalResult) -> ErrorCategory:
        """Categorize the type of error.

        Args:
            result: The evaluation result

        Returns:
            Error category
        """
        expected = result.case.expected_output.strip().lower()
        actual = result.actual_output.strip().lower()

        # Check for format violations (extra text)
        if expected in actual and len(actual) > len(expected) * 1.5:
            return ErrorCategory.TOO_VERBOSE

        # Check for missing information
        if len(actual) < len(expected) * 0.5:
            return ErrorCategory.MISSING_INFO

        # Check for format violations (wrong structure)
        if " " in expected and " " not in actual:
            return ErrorCategory.FORMAT_VIOLATION

        # For simple classification tasks, assume boundary confusion
        # if outputs are similar categories
        if self._are_similar_categories(expected, actual):
            return ErrorCategory.BOUNDARY_CONFUSION

        # Default to logic error
        return ErrorCategory.LOGIC_ERROR

    def _are_similar_categories(self, expected: str, actual: str) -> bool:
        """Check if two outputs represent similar categories.

        Args:
            expected: Expected output
            actual: Actual output

        Returns:
            True if they're similar categories (e.g., positive vs neutral)
        """
        # Common similar pairs
        similar_pairs = [
            {"positive", "neutral"},
            {"negative", "neutral"},
            {"true", "false"},
            {"yes", "no"},
        ]

        expected_lower = expected.lower()
        actual_lower = actual.lower()

        for pair in similar_pairs:
            if expected_lower in pair and actual_lower in pair:
                return True

        return False

    def _generate_analysis(self, result: EvalResult, category: ErrorCategory) -> str:
        """Generate human-readable analysis of the error.

        Args:
            result: The evaluation result
            category: The error category

        Returns:
            Analysis text
        """
        analyses = {
            ErrorCategory.FORMAT_VIOLATION: (
                "Output format doesn't match expected format. "
                "Instructions may need to be more explicit about output structure."
            ),
            ErrorCategory.TOO_VERBOSE: (
                "Output contains additional text beyond what was requested. "
                "Instructions should emphasize outputting ONLY the required information."
            ),
            ErrorCategory.MISSING_INFO: (
                "Output is missing required information. "
                "Instructions should be clearer about what information must be included."
            ),
            ErrorCategory.BOUNDARY_CONFUSION: (
                "Ambiguous case classified incorrectly. "
                "Instructions need clearer guidance for edge cases and boundary conditions."
            ),
            ErrorCategory.LOGIC_ERROR: (
                "Incorrect reasoning or interpretation of the input. "
                "Instructions may need more examples or clearer task definition."
            ),
            ErrorCategory.OTHER: "Unspecified error. Review the diff for details.",
        }

        return analyses.get(category, analyses[ErrorCategory.OTHER])
