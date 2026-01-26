"""Test runner for evaluating prompts against eval cases."""

import sys
import threading
import time

from src.optimizer.models import EvalCase, EvalResult
from src.providers.base import LLMProvider


class Spinner:
    """Simple loading spinner for terminal."""

    def __init__(self, message: str = "Loading"):
        self.message = message
        self.running = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.frame_idx = 0

    def start(self):
        """Start the spinner in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        """Run the spinner animation."""
        while self.running:
            frame = self.frames[self.frame_idx % len(self.frames)]
            sys.stdout.write(f"\r{frame} {self.message}")
            sys.stdout.flush()
            self.frame_idx += 1
            time.sleep(0.1)

    def stop(self, final_message: str = ""):
        """Stop the spinner and clear the line."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        if final_message:
            sys.stdout.write(final_message + "\n")
        sys.stdout.flush()


class TestRunner:
    """Executes prompts against evaluation cases."""

    def __init__(self, target_provider: LLMProvider):
        """Initialize the test runner.

        Args:
            target_provider: LLM provider to test prompts with (e.g., Sonnet)
        """
        self.target_provider = target_provider

    def run_eval(
        self,
        prompt: str,
        eval_cases: list[EvalCase],
        system: str | None = None,
    ) -> list[EvalResult]:
        """Run prompt against all eval cases.

        Args:
            prompt: The prompt to evaluate
            eval_cases: List of evaluation cases
            system: Optional system message

        Returns:
            List of evaluation results
        """
        results = []
        total_cases = len(eval_cases)

        for idx, case in enumerate(eval_cases, 1):
            # Start spinner for this case
            spinner = Spinner(f"Evaluating case {idx}/{total_cases}")
            spinner.start()

            try:
                # Generate the full prompt by combining the instruction prompt
                # with the specific input
                full_prompt = f"{prompt}\n\nInput: {case.input}"

                # Generate output using target model
                actual_output = self.target_provider.generate(
                    prompt=full_prompt,
                    system=system,
                    temperature=0.0,  # Deterministic for evaluation
                    max_tokens=500,
                )

                # Check if output matches expected
                passed = self._check_match(actual_output, case.expected_output)

                # Stop spinner and show result
                status = "✓" if passed else "✗"
                spinner.stop(f"  {status} Case {idx}/{total_cases}")

                results.append(
                    EvalResult(
                        case=case,
                        actual_output=actual_output,
                        passed=passed,
                        error=None,
                    )
                )

            except Exception as e:
                # Stop spinner and show error
                spinner.stop(f"  ✗ Case {idx}/{total_cases} (error)")

                # Handle errors during execution
                results.append(
                    EvalResult(
                        case=case,
                        actual_output="",
                        passed=False,
                        error=str(e),
                    )
                )

        return results

    def _check_match(self, actual: str, expected: str) -> bool:
        """Check if actual output matches expected output.

        Args:
            actual: The actual output from the model
            expected: The expected output

        Returns:
            True if they match (after normalization)
        """
        # Normalize both strings for comparison
        actual_normalized = actual.strip().lower()
        expected_normalized = expected.strip().lower()

        return actual_normalized == expected_normalized

    def compute_pass_rate(self, results: list[EvalResult]) -> tuple[float, int, int]:
        """Compute pass rate from results.

        Args:
            results: List of evaluation results

        Returns:
            Tuple of (pass_rate, passed_count, total_count)
        """
        if not results:
            return 0.0, 0, 0

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0.0

        return pass_rate, passed, total
