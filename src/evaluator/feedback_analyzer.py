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
        """Extract a specific error pattern from multiple failure cases.

        This generates descriptive feedback WITHOUT revealing specific test inputs/outputs,
        but analyzes actual failures to provide concrete patterns.

        Args:
            category: The error category
            cases: List of failures in this category

        Returns:
            Error pattern with specific descriptions based on actual failures
        """
        # Analyze actual failures to extract patterns
        expected_outputs = [c.case.expected_output.strip().lower() for c in cases]
        actual_outputs = [c.actual_output.strip().lower() for c in cases]

        # Find common patterns in the failures
        common_expected = self._find_common_patterns(expected_outputs)
        common_actual = self._find_common_patterns(actual_outputs)

        # Generate more specific pattern descriptions
        pattern_observed = self._describe_pattern_observed(category, cases, expected_outputs, actual_outputs)
        example_pattern = self._describe_example_pattern(category, common_expected, common_actual)
        root_cause = self._describe_root_cause(category, cases)
        recommended_fix = self._describe_recommended_fix(category, common_expected, common_actual, cases)

        return ErrorPattern(
            error_type=category,
            count=len(cases),
            pattern_observed=pattern_observed,
            example_pattern=example_pattern,
            root_cause=root_cause,
            recommended_fix=recommended_fix,
        )

    def _find_common_patterns(self, outputs: list[str]) -> dict:
        """Find common patterns in a list of outputs.

        Args:
            outputs: List of output strings

        Returns:
            Dict with pattern information
        """
        if not outputs:
            return {}

        # Check for common prefixes/suffixes
        common_prefix = ""
        common_suffix = ""

        # Find common words/categories
        words = set()
        for output in outputs:
            words.update(output.split())

        return {
            "unique_values": len(set(outputs)),
            "common_words": list(words)[:5],  # Top 5 most common words
            "avg_length": sum(len(o) for o in outputs) / len(outputs),
        }

    def _describe_pattern_observed(
        self, category: ErrorCategory, cases: list[EvalResult],
        expected: list[str], actual: list[str]
    ) -> str:
        """Describe what pattern was observed in the failures."""

        # Analyze specific patterns based on category
        if category == ErrorCategory.FORMAT_VIOLATION:
            avg_expected_len = sum(len(e) for e in expected) / len(expected)
            avg_actual_len = sum(len(a) for a in actual) / len(actual)
            return (
                f"Output format deviates from expected structure. "
                f"Expected outputs average {avg_expected_len:.0f} chars, "
                f"actual outputs average {avg_actual_len:.0f} chars."
            )

        elif category == ErrorCategory.TOO_VERBOSE:
            extra_words_count = sum(len(a.split()) - len(e.split()) for a, e in zip(actual, expected)) / len(actual)
            return (
                f"Outputs include {extra_words_count:.1f} extra words on average beyond required content. "
                f"Adding explanations, reasoning, or additional commentary."
            )

        elif category == ErrorCategory.MISSING_INFO:
            missing_ratio = sum(len(a) / len(e) if len(e) > 0 else 0 for a, e in zip(actual, expected)) / len(actual)
            return (
                f"Outputs are incomplete, containing only {missing_ratio:.0%} of expected information on average. "
                f"Missing required components or details."
            )

        elif category == ErrorCategory.BOUNDARY_CONFUSION:
            unique_expected = len(set(expected))
            unique_actual = len(set(actual))
            return (
                f"Ambiguous cases misclassified. {unique_expected} different expected categories "
                f"confused with {unique_actual} different actual outputs. "
                f"Edge cases and boundary conditions handled incorrectly."
            )

        elif category == ErrorCategory.LOGIC_ERROR:
            return (
                f"Fundamental reasoning errors in {len(cases)} cases. "
                f"Model interpretation differs from intended task semantics."
            )

        else:
            return f"Unspecified error pattern affecting {len(cases)} cases"

    def _describe_example_pattern(self, category: ErrorCategory, common_expected: dict, common_actual: dict) -> str:
        """Describe an example of the error pattern."""

        if category == ErrorCategory.FORMAT_VIOLATION:
            return (
                f"Expected format typically uses {common_expected.get('avg_length', 0):.0f} characters. "
                f"Actual outputs use different structure or formatting."
            )

        elif category == ErrorCategory.TOO_VERBOSE:
            return (
                "Outputs contain phrases like explanations, justifications, or additional context "
                "beyond the required classification/answer."
            )

        elif category == ErrorCategory.MISSING_INFO:
            expected_words = common_expected.get('common_words', [])
            return (
                f"Expected outputs typically include elements like: {', '.join(expected_words[:3])}. "
                f"Actual outputs are missing these components."
            )

        elif category == ErrorCategory.BOUNDARY_CONFUSION:
            expected_words = common_expected.get('common_words', [])
            actual_words = common_actual.get('common_words', [])
            return (
                f"Cases requiring distinction between categories like '{', '.join(expected_words[:2])}' "
                f"are being classified as '{', '.join(actual_words[:2])}' instead."
            )

        elif category == ErrorCategory.LOGIC_ERROR:
            return "Model's reasoning about input semantics differs from intended interpretation."

        else:
            return "Error type doesn't fit standard categories"

    def _describe_root_cause(self, category: ErrorCategory, cases: list[EvalResult]) -> str:
        """Describe the likely root cause of the error."""

        base_patterns = {
            ErrorCategory.FORMAT_VIOLATION: "Instructions not explicit enough about output format requirements",
            ErrorCategory.TOO_VERBOSE: "Instructions don't emphasize that ONLY the answer should be provided",
            ErrorCategory.MISSING_INFO: "Instructions unclear about all required output components",
            ErrorCategory.BOUNDARY_CONFUSION: (
                f"Instructions lack clear guidance for resolving {len(cases)} ambiguous cases"
            ),
            ErrorCategory.LOGIC_ERROR: "Task description or examples don't cover this type of reasoning",
            ErrorCategory.OTHER: "Unable to determine root cause from error patterns",
        }

        return base_patterns.get(category, base_patterns[ErrorCategory.OTHER])

    def _describe_recommended_fix(
        self, category: ErrorCategory, common_expected: dict, common_actual: dict, cases: list[EvalResult]
    ) -> str:
        """Describe a recommended fix for the error."""

        if category == ErrorCategory.FORMAT_VIOLATION:
            return (
                "Add explicit format examples showing EXACTLY what the output should look like. "
                "Use phrases like 'Output must match this exact format:' with concrete examples."
            )

        elif category == ErrorCategory.TOO_VERBOSE:
            return (
                "Add strong emphasis: 'Output ONLY the classification/answer. "
                "Do NOT include explanations, reasoning, justifications, or any additional text.' "
                "Consider adding: 'Your entire response should be a single word/phrase.'"
            )

        elif category == ErrorCategory.MISSING_INFO:
            expected_words = common_expected.get('common_words', [])
            return (
                f"Explicitly list all required components. For example: "
                f"'Every output must include: {', '.join(expected_words[:3])}...' "
                f"Use a checklist format to make requirements crystal clear."
            )

        elif category == ErrorCategory.BOUNDARY_CONFUSION:
            return (
                f"Add {len(cases)} specific rules for handling ambiguous cases. "
                f"Define clear decision criteria: 'When inputs have mixed signals, prioritize X over Y.' "
                f"Provide explicit guidance on edge cases and tie-breaking."
            )

        elif category == ErrorCategory.LOGIC_ERROR:
            return (
                f"Clarify task semantics with more explicit reasoning guidelines. "
                f"Add {min(3, len(cases))} examples demonstrating the correct interpretation for similar cases."
            )

        else:
            return "Review instructions for clarity and completeness"

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
