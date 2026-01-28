"""Test runner for evaluating prompts against eval cases."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    _ALLOWED_CATEGORIES = {
        "RETURN_REQUEST",
        "TRACKING_INQUIRY",
        "PRODUCT_INQUIRY",
        "BILLING_ISSUE",
        "COMPLAINT",
        "ORDER_MODIFICATION",
        "POSITIVE_FEEDBACK",
        "REFUND_REQUEST",
        "POLICY_QUESTION",
        "DISCOUNT_INQUIRY",
    }
    _ALLOWED_DETAILS = {
        "ORDER_STATUS",
        "SIZE",
        "COLOR",
        "MATERIAL",
        "FEATURES",
        "DUPLICATE_CHARGE",
        "UNAUTHORIZED_CHARGE",
        "PRICE_DISCREPANCY",
        "INCORRECT_AMOUNT",
        "PRODUCT_DEFECT",
        "POOR_QUALITY",
        "MISLEADING_DESCRIPTION",
        "SERVICE_ISSUE",
        "SHIPPING_DAMAGE",
        "ORDER_CANCELLATION",
        "ADD_ITEMS",
        "REMOVE_ITEMS",
        "ADDRESS_CHANGE",
        "REFUND_DAMAGED",
        "REFUND_DEFECTIVE",
        "REFUND_NOT_RECEIVED",
        "REFUND_QUALITY",
        "REFUND_GENERAL",
        "RETURN_POLICY",
        "RETURN_WINDOW",
        "WARRANTY_POLICY",
        "SHIPPING_POLICY",
        "PRICE_MATCHING",
        "PROMO_CODE_REQUEST",
        "CURRENT_PROMOTIONS",
        "COUPON_AVAILABILITY",
        "STUDENT_DISCOUNT",
        "NEW_CUSTOMER_DISCOUNT",
        "POSITIVE_GENERAL",
        "PRODUCT_MISMATCH",
        "RETURN_GENERAL",
    }
    _ALLOWED_CHANGES = {"ADDRESS", "COLOR", "ITEMS_ADD", "ITEMS_REMOVE", "CANCEL", "NONE"}

    def __init__(self, target_provider: LLMProvider, max_workers: int = 10):
        """Initialize the test runner.

        Args:
            target_provider: LLM provider to test prompts with (e.g., Sonnet)
            max_workers: Maximum number of parallel API calls (default: 10)
        """
        self.target_provider = target_provider
        self.max_workers = max_workers
        self._cache = {}  # Cache for (prompt, input, system) -> output
        self._cache_lock = threading.Lock()  # Thread-safe cache access
        self.cache_hits = 0
        self.cache_misses = 0

    def run_eval(
        self,
        prompt: str,
        eval_cases: list[EvalCase],
        system: str | None = None,
    ) -> list[EvalResult]:
        """Run prompt against all eval cases in parallel.

        Args:
            prompt: The prompt to evaluate
            eval_cases: List of evaluation cases
            system: Optional system message

        Returns:
            List of evaluation results (in original order)
        """
        total_cases = len(eval_cases)
        completed = 0
        passed_count = 0
        failed_count = 0
        lock = threading.Lock()

        # Progress tracker with spinner
        progress_message = f"Evaluating: 0/{total_cases} completed (0 passed, 0 failed)"
        spinner = Spinner(progress_message)
        spinner.start()

        def update_progress():
            """Update the spinner message with current progress."""
            nonlocal progress_message
            progress_message = f"Evaluating: {completed}/{total_cases} completed ({passed_count} passed, {failed_count} failed)"
            spinner.message = progress_message

        # Store results with their original index
        results_dict = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._evaluate_single_case, prompt, case, system): idx
                for idx, case in enumerate(eval_cases)
            }

            # Process completed tasks
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                results_dict[idx] = result

                # Update counters
                with lock:
                    completed += 1
                    if result.passed:
                        passed_count += 1
                    else:
                        failed_count += 1
                    update_progress()

        # Stop spinner and show final results
        spinner.stop(f"  Completed: {total_cases}/{total_cases} ({passed_count} passed, {failed_count} failed)")

        # Return results in original order
        return [results_dict[i] for i in range(len(eval_cases))]

    def _evaluate_single_case(
        self,
        prompt: str,
        case: EvalCase,
        system: str | None = None,
    ) -> EvalResult:
        """Evaluate a single case (used for parallel execution).

        Args:
            prompt: The prompt to evaluate
            case: The evaluation case
            system: Optional system message

        Returns:
            Evaluation result
        """
        # Check cache first (thread-safe)
        cache_key = (prompt, case.input, system)
        with self._cache_lock:
            if cache_key in self._cache:
                self.cache_hits += 1
                actual_output = self._cache[cache_key]

                # Check if output matches expected
                passed = self._check_match(actual_output, case.expected_output)

                return EvalResult(
                    case=case,
                    actual_output=actual_output,
                    passed=passed,
                    error=None,
                )

            self.cache_misses += 1

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

            # Store in cache (thread-safe)
            with self._cache_lock:
                self._cache[cache_key] = actual_output

            # Check if output matches expected
            passed = self._check_match(actual_output, case.expected_output)

            return EvalResult(
                case=case,
                actual_output=actual_output,
                passed=passed,
                error=None,
            )

        except Exception as e:
            # Handle errors during execution
            return EvalResult(
                case=case,
                actual_output="",
                passed=False,
                error=str(e),
            )

    def _check_match(self, actual: str, expected: str) -> bool:
        """Check if actual output matches expected output.

        Args:
            actual: The actual output from the model
            expected: The expected output

        Returns:
            True if they match (after normalization)
        """
        actual_parsed = self._parse_labeled_output(actual)
        expected_parsed = self._parse_labeled_output(expected)
        if not actual_parsed or not expected_parsed:
            return False
        return actual_parsed == expected_parsed

    def _parse_labeled_output(self, output: str) -> dict[str, str] | None:
        parts = [part.strip() for part in output.split(";") if part.strip()]
        if len(parts) != 4:
            return None

        parsed: dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                return None
            key, value = part.split("=", 1)
            key = key.strip().upper()
            value = value.strip().upper()
            if not key or not value or key in parsed:
                return None
            parsed[key] = value

        if set(parsed.keys()) != {"CATEGORY", "DETAIL", "ORDER_ID", "CHANGE"}:
            return None
        if parsed["CATEGORY"] not in self._ALLOWED_CATEGORIES:
            return None
        if parsed["DETAIL"] not in self._ALLOWED_DETAILS:
            return None
        if parsed["CHANGE"] not in self._ALLOWED_CHANGES:
            return None

        order_id = parsed["ORDER_ID"]
        if order_id != "NONE" and not order_id.isdigit():
            return None
        if parsed["CATEGORY"] != "ORDER_MODIFICATION" and parsed["CHANGE"] != "NONE":
            return None

        return parsed

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

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
