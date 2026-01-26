"""Main optimization loop orchestrator."""

from src.evaluator.feedback_analyzer import FeedbackAnalyzer
from src.evaluator.test_runner import TestRunner
from src.optimizer.meta_optimizer import MetaOptimizer
from src.optimizer.models import (
    EvalCase,
    IterationResult,
    OptimizationConfig,
    OptimizationResult,
    OptimizationStatus,
)
from src.orchestrator.prompt_history import PromptHistory
from src.providers.base import LLMProvider


class OptimizationLoop:
    """Orchestrates the optimization process."""

    def __init__(
        self,
        config: OptimizationConfig,
        optimizer_provider: LLMProvider,
        target_provider: LLMProvider,
    ):
        """Initialize the optimization loop.

        Args:
            config: Optimization configuration
            optimizer_provider: Provider for the meta-optimizer (e.g., Opus)
            target_provider: Provider for the target model being optimized
        """
        self.config = config
        self.meta_optimizer = MetaOptimizer(optimizer_provider)
        self.test_runner = TestRunner(target_provider, max_workers=config.optimizer.max_workers)
        self.feedback_analyzer = FeedbackAnalyzer()
        self.iteration_history: list[IterationResult] = []
        self.prompt_history = PromptHistory()

    def optimize(
        self,
        training_cases: list[EvalCase],
        test_cases: list[EvalCase] | None = None,
    ) -> OptimizationResult:
        """Run the two-phase optimization loop.

        Phase 1: Optimize on training set with full feedback
        Phase 2: Validate on test set with descriptive feedback

        Args:
            training_cases: Training evaluation cases
            test_cases: Test evaluation cases (optional for MVP Phase 1 only)

        Returns:
            Optimization result
        """
        print(f"Starting optimization with {len(training_cases)} training cases", end="")
        if test_cases:
            print(f" and {len(test_cases)} test cases")
        else:
            print()
        print(f"Target model: {self.test_runner.target_provider.model_name}")
        print(f"Optimizer: {self.meta_optimizer.provider.model_name}\n")

        # Step 1: Generate initial prompt
        print("\033[94mIteration 0: Generating initial prompt...\033[0m")
        current_prompt = self.meta_optimizer.generate_initial_prompt(
            task_description=self.config.task_description,
            training_examples=training_cases,
        )
        print(f"Initial prompt:\n{current_prompt}\n")

        # Step 2: Evaluate initial prompt
        results = self.test_runner.run_eval(current_prompt, training_cases)
        pass_rate, passed, total = self.test_runner.compute_pass_rate(results)
        print(f"Initial evaluation: {passed}/{total} passed ({pass_rate:.1%})\n")

        # Record iteration 0
        self.iteration_history.append(
            IterationResult(
                iteration=0,
                prompt=current_prompt,
                training_pass_rate=pass_rate,
                optimizer_tokens_used=self.meta_optimizer.get_total_tokens_used(),
            )
        )

        # Add to prompt history
        self.prompt_history.add_version(
            iteration=0,
            prompt=current_prompt,
            training_score=pass_rate,
        )

        # Step 3: Phase 1 - Training optimization loop
        print("=" * 70)
        print("PHASE 1: Training Set Optimization")
        print("=" * 70)
        print()

        iteration = 1
        training_pass_rate = pass_rate
        while iteration <= self.config.optimizer.max_iterations:
            # Check if we've reached 100% on training
            if training_pass_rate >= self.config.stopping_criteria.training_pass_rate:
                print(f"✓ Training set: 100% pass rate achieved!\n")

                # If we have test cases, proceed to Phase 2
                if test_cases:
                    return self._optimize_test_phase(
                        current_prompt=current_prompt,
                        test_cases=test_cases,
                        starting_iteration=iteration,
                        training_pass_rate=training_pass_rate,
                    )
                else:
                    # No test set - return success with training only
                    return self._create_result(
                        status=OptimizationStatus.SUCCESS,
                        final_prompt=current_prompt,
                        training_pass_rate=training_pass_rate,
                        test_pass_rate=0.0,
                    )

            # Check token budget
            if (
                self.meta_optimizer.get_total_tokens_used()
                >= self.config.optimizer.max_optimizer_tokens
            ):
                print("⚠ Token budget exceeded\n")
                return self._create_result(
                    status=OptimizationStatus.TOKEN_BUDGET_EXCEEDED,
                    final_prompt=current_prompt,
                    training_pass_rate=pass_rate,
                )

            # Generate detailed feedback for failures
            feedback = self.feedback_analyzer.analyze_training_results(results)

            print(f"\033[94mIteration {iteration}: Refining prompt...\033[0m")
            print(f"  Pass rate: {passed}/{total} ({pass_rate:.1%})")
            print(f"  Failures: {len(feedback.failures)}")

            # Save current prompt before refinement
            previous_prompt = current_prompt
            previous_pass_rate = training_pass_rate

            # Refine prompt based on feedback
            candidate_prompt = self.meta_optimizer.refine_prompt_training(
                current_prompt=current_prompt,
                feedback=feedback,
            )
            print(f"\n  Candidate prompt:\n{candidate_prompt}\n")

            # Evaluate refined prompt
            results = self.test_runner.run_eval(candidate_prompt, training_cases)
            new_pass_rate, passed, total = self.test_runner.compute_pass_rate(results)

            # Validate: only accept if improvement or maintaining performance
            if self.prompt_history.should_accept(new_pass_rate, previous_pass_rate, phase="training"):
                current_prompt = candidate_prompt
                training_pass_rate = new_pass_rate
                print(f"  \033[92m✓ Accepted: {previous_pass_rate:.1%} → {new_pass_rate:.1%}\033[0m")
            else:
                # Reject and keep previous prompt
                current_prompt = previous_prompt
                training_pass_rate = previous_pass_rate
                print(f"  \033[91m✗ Rejected (regression): {previous_pass_rate:.1%} → {new_pass_rate:.1%}\033[0m")
                print(f"  Keeping previous prompt")

            print(f"  Current pass rate: {int(training_pass_rate * total)}/{total} ({training_pass_rate:.1%})\n")

            # Add to prompt history
            self.prompt_history.add_version(
                iteration=iteration,
                prompt=current_prompt,
                training_score=training_pass_rate,
            )

            # Record iteration
            self.iteration_history.append(
                IterationResult(
                    iteration=iteration,
                    prompt=current_prompt,
                    training_pass_rate=training_pass_rate,
                    feedback=feedback,
                    optimizer_tokens_used=self.meta_optimizer.get_total_tokens_used(),
                )
            )

            # Check for plateau
            if self._is_plateaued():
                print("⚠ Optimization plateaued (no improvement)\n")
                return self._create_result(
                    status=OptimizationStatus.PLATEAU_DETECTED,
                    final_prompt=current_prompt,
                    training_pass_rate=training_pass_rate,
                    test_pass_rate=0.0,
                )

            iteration += 1

        # Max iterations reached
        print("⚠ Maximum iterations reached\n")
        return self._create_result(
            status=OptimizationStatus.MAX_ITERATIONS,
            final_prompt=current_prompt,
            training_pass_rate=training_pass_rate,
            test_pass_rate=0.0,
        )

    def _optimize_test_phase(
        self,
        current_prompt: str,
        test_cases: list[EvalCase],
        starting_iteration: int,
        training_pass_rate: float,
    ) -> OptimizationResult:
        """Phase 2: Optimize on test set with descriptive feedback.

        Args:
            current_prompt: The prompt that passed training
            test_cases: Test evaluation cases
            starting_iteration: Iteration number to start from
            training_pass_rate: Final training pass rate

        Returns:
            Optimization result
        """
        print("=" * 70)
        print("PHASE 2: Test Set Validation")
        print("=" * 70)
        print()

        # Evaluate on test set
        results = self.test_runner.run_eval(current_prompt, test_cases)
        test_pass_rate, passed, total = self.test_runner.compute_pass_rate(results)
        print(f"Initial test evaluation: {passed}/{total} passed ({test_pass_rate:.1%})\n")

        # Record initial test evaluation
        self.iteration_history.append(
            IterationResult(
                iteration=starting_iteration,
                prompt=current_prompt,
                training_pass_rate=training_pass_rate,
                test_pass_rate=test_pass_rate,
                optimizer_tokens_used=self.meta_optimizer.get_total_tokens_used(),
            )
        )

        # Update prompt history with test score
        self.prompt_history.add_version(
            iteration=starting_iteration,
            prompt=current_prompt,
            training_score=training_pass_rate,
            test_score=test_pass_rate,
        )

        test_iterations = 0
        iteration = starting_iteration + 1

        # Phase 2 loop: Refine based on test feedback
        while test_iterations < self.config.optimizer.max_test_iterations:
            # Check if we've reached 100% on test
            if test_pass_rate >= self.config.stopping_criteria.test_pass_rate:
                print(f"✓ Test set: 100% pass rate achieved!\n")
                return self._create_result(
                    status=OptimizationStatus.SUCCESS,
                    final_prompt=current_prompt,
                    training_pass_rate=training_pass_rate,
                    test_pass_rate=test_pass_rate,
                )

            # Check token budget
            if (
                self.meta_optimizer.get_total_tokens_used()
                >= self.config.optimizer.max_optimizer_tokens
            ):
                print("⚠ Token budget exceeded\n")
                return self._create_result(
                    status=OptimizationStatus.TOKEN_BUDGET_EXCEEDED,
                    final_prompt=current_prompt,
                    training_pass_rate=training_pass_rate,
                    test_pass_rate=test_pass_rate,
                )

            # Check overall iteration limit
            if iteration > self.config.optimizer.max_iterations:
                print("⚠ Maximum total iterations reached\n")
                return self._create_result(
                    status=OptimizationStatus.MAX_ITERATIONS,
                    final_prompt=current_prompt,
                    training_pass_rate=training_pass_rate,
                    test_pass_rate=test_pass_rate,
                )

            # Generate descriptive feedback (patterns only, no specifics)
            feedback = self.feedback_analyzer.analyze_test_results(results)

            print(f"\033[94mIteration {iteration}: Refining based on test patterns...\033[0m")
            print(f"  Test pass rate: {passed}/{total} ({test_pass_rate:.1%})")
            print(f"  Error patterns: {len(feedback.error_patterns)}")

            # Save current prompt before refinement
            previous_prompt = current_prompt
            previous_test_pass_rate = test_pass_rate

            # Refine prompt based on descriptive test feedback
            candidate_prompt = self.meta_optimizer.refine_prompt_test(
                current_prompt=current_prompt,
                feedback=feedback,
            )
            print(f"\n  Candidate prompt:\n{candidate_prompt}\n")

            # Evaluate refined prompt on test set
            results = self.test_runner.run_eval(candidate_prompt, test_cases)
            new_test_pass_rate, passed, total = self.test_runner.compute_pass_rate(results)

            # Validate: only accept if improvement or maintaining performance
            if self.prompt_history.should_accept(new_test_pass_rate, previous_test_pass_rate, phase="test"):
                current_prompt = candidate_prompt
                test_pass_rate = new_test_pass_rate
                print(f"  \033[92m✓ Accepted: {previous_test_pass_rate:.1%} → {new_test_pass_rate:.1%}\033[0m")
            else:
                # Reject and keep previous prompt
                current_prompt = previous_prompt
                test_pass_rate = previous_test_pass_rate
                print(f"  \033[91m✗ Rejected (regression): {previous_test_pass_rate:.1%} → {new_test_pass_rate:.1%}\033[0m")
                print(f"  Keeping previous prompt")

            print(f"  Current test pass rate: {int(test_pass_rate * total)}/{total} ({test_pass_rate:.1%})\n")

            # Add to prompt history
            self.prompt_history.add_version(
                iteration=iteration,
                prompt=current_prompt,
                training_score=training_pass_rate,
                test_score=test_pass_rate,
            )

            # Record iteration
            self.iteration_history.append(
                IterationResult(
                    iteration=iteration,
                    prompt=current_prompt,
                    training_pass_rate=training_pass_rate,
                    test_pass_rate=test_pass_rate,
                    feedback=feedback,
                    optimizer_tokens_used=self.meta_optimizer.get_total_tokens_used(),
                )
            )

            test_iterations += 1
            iteration += 1

        # Test iteration limit reached
        print("⚠ Test iteration limit reached\n")
        return self._create_result(
            status=OptimizationStatus.MAX_ITERATIONS,
            final_prompt=current_prompt,
            training_pass_rate=training_pass_rate,
            test_pass_rate=test_pass_rate,
        )

    def _is_plateaued(self) -> bool:
        """Check if optimization has plateaued.

        Returns:
            True if no improvement for plateau_threshold iterations
        """
        threshold = self.config.optimizer.plateau_threshold

        if len(self.iteration_history) < threshold:
            return False

        # Get recent pass rates
        recent = self.iteration_history[-threshold:]
        pass_rates = [
            r.training_pass_rate for r in recent if r.training_pass_rate is not None
        ]

        if len(pass_rates) < threshold:
            return False

        # Check if all pass rates are the same (no improvement)
        return len(set(pass_rates)) == 1

    def _create_result(
        self,
        status: OptimizationStatus,
        final_prompt: str,
        training_pass_rate: float,
        test_pass_rate: float,
    ) -> OptimizationResult:
        """Create optimization result.

        Args:
            status: Optimization status
            final_prompt: The final optimized prompt
            training_pass_rate: Final training pass rate
            test_pass_rate: Final test pass rate

        Returns:
            Optimization result
        """
        # Create appropriate status message
        if status == OptimizationStatus.SUCCESS:
            if test_pass_rate > 0:
                message = (
                    f"Successfully reached 100% on both training and test sets! "
                    f"Training: {training_pass_rate:.1%}, Test: {test_pass_rate:.1%}"
                )
            else:
                message = f"Successfully reached 100% on training set!"
        elif status == OptimizationStatus.MAX_ITERATIONS:
            message = (
                f"Reached maximum iterations ({self.config.optimizer.max_iterations}). "
                f"Training: {training_pass_rate:.1%}, Test: {test_pass_rate:.1%}"
            )
        elif status == OptimizationStatus.TOKEN_BUDGET_EXCEEDED:
            message = (
                f"Token budget exceeded. "
                f"Training: {training_pass_rate:.1%}, Test: {test_pass_rate:.1%}"
            )
        elif status == OptimizationStatus.PLATEAU_DETECTED:
            message = (
                f"Optimization plateaued. "
                f"Training: {training_pass_rate:.1%}, Test: {test_pass_rate:.1%}"
            )
        else:
            message = "Unknown status"

        return OptimizationResult(
            status=status,
            final_prompt=final_prompt,
            iterations=self.iteration_history,
            training_pass_rate=training_pass_rate,
            test_pass_rate=test_pass_rate,
            total_optimizer_tokens=self.meta_optimizer.get_total_tokens_used(),
            message=message,
        )
