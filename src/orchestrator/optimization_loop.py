"""Main optimization loop orchestrator."""

from src.evaluator.feedback_analyzer import FeedbackAnalyzer
from src.evaluator.test_runner import TestRunner
from src.optimizer.meta_optimizer import MetaOptimizer
from src.optimizer.models import (
    CombinedFeedback,
    EvalCase,
    IterationResult,
    OptimizationConfig,
    OptimizationResult,
    OptimizationStatus,
)
from src.optimizer.optimization_memory import OptimizationMemory, IterationSummary
from src.orchestrator.prompt_history import PromptHistory
from src.providers.base import LLMProvider


class OptimizationLoop:
    """Orchestrates the optimization process."""

    def __init__(
        self,
        config: OptimizationConfig,
        optimizer_provider: LLMProvider,
        target_provider: LLMProvider,
        verbose: bool = False,
    ):
        """Initialize the optimization loop.

        Args:
            config: Optimization configuration
            optimizer_provider: Provider for the meta-optimizer (e.g., Opus)
            target_provider: Provider for the target model being optimized
            verbose: Show refined prompts at each iteration
        """
        self.config = config
        self.meta_optimizer = MetaOptimizer(optimizer_provider)
        self.test_runner = TestRunner(
            target_provider,
            max_workers=config.optimizer.max_workers,
            schema=config.schema,
        )
        self.feedback_analyzer = FeedbackAnalyzer()
        self.iteration_history: list[IterationResult] = []
        self.prompt_history = PromptHistory()
        self.optimization_memory = OptimizationMemory(max_recent_history=3)
        self.verbose = verbose

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
                        training_cases=training_cases,
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

            # Get current pass rate for display
            current_pass_rate, current_passed, current_total = self.test_runner.compute_pass_rate(results)

            print(f"\033[94mIteration {iteration}: Refining prompt...\033[0m")
            print(f"  Pass rate: {current_passed}/{current_total} ({current_pass_rate:.1%})")
            print(f"  Failures: {len(feedback.failures)}")

            # Save current state before refinement
            previous_results = results
            previous_prompt = current_prompt
            previous_pass_rate = training_pass_rate

            # Generate multiple candidate prompts
            optimization_context = self.optimization_memory.get_context_string()
            num_candidates = self.config.optimizer.candidates_per_iteration

            print(f"  Generating {num_candidates} candidate prompts...")
            candidates = self.meta_optimizer.generate_multiple_candidates(
                base_method='training',
                num_candidates=num_candidates,
                current_prompt=current_prompt,
                feedback=feedback,
                optimization_context=optimization_context,
            )

            # Evaluate all candidates and pick the best
            best_candidate = None
            best_results = None
            best_pass_rate = previous_pass_rate  # Need to beat current

            for idx, candidate_prompt in enumerate(candidates, 1):
                print(f"  Evaluating candidate {idx}/{num_candidates}...")
                if self.verbose:
                    print(f"    Candidate {idx} prompt:\n{candidate_prompt}\n")

                # Evaluate this candidate
                candidate_results = self.test_runner.run_eval(candidate_prompt, training_cases)
                candidate_pass_rate, candidate_passed, candidate_total = self.test_runner.compute_pass_rate(candidate_results)
                print(f"    → {candidate_passed}/{candidate_total} ({candidate_pass_rate:.1%})")

                # Track best candidate (use >= to allow lateral moves, matching acceptance logic)
                if candidate_pass_rate >= best_pass_rate:
                    best_pass_rate = candidate_pass_rate
                    best_candidate = candidate_prompt
                    best_results = candidate_results

            # Use the best candidate (with >=, at least one candidate will match current score)
            if best_candidate is not None:
                candidate_prompt = best_candidate
                candidate_results = best_results
                new_pass_rate = best_pass_rate
            else:
                # This should never happen with >= comparison, but handle gracefully
                # No candidate matched or exceeded current - reject all
                candidate_prompt = previous_prompt
                candidate_results = previous_results
                new_pass_rate = previous_pass_rate

            # Validate: only accept if improvement
            accepted = self.prompt_history.should_accept(new_pass_rate, previous_pass_rate, phase="training")

            if accepted:
                # Accept: update to new prompt and results
                results = candidate_results
                current_prompt = candidate_prompt
                training_pass_rate = new_pass_rate
                print(f"  \033[92m✓ Accepted: {previous_pass_rate:.1%} → {new_pass_rate:.1%}\033[0m")
            else:
                # Reject and restore previous state
                results = previous_results  # Restore results to match current_prompt
                current_prompt = previous_prompt
                training_pass_rate = previous_pass_rate
                print(f"  \033[91m✗ Rejected (regression): {previous_pass_rate:.1%} → {new_pass_rate:.1%}\033[0m")
                print(f"  Keeping previous prompt")

            print(f"  Current pass rate: {int(training_pass_rate * total)}/{total} ({training_pass_rate:.1%})\n")

            # Record iteration summary in memory
            target_issues = self._extract_target_issues(feedback)
            prompt_change_desc = self._describe_prompt_change(feedback, phase="training")
            outcome = self._describe_outcome(previous_pass_rate, new_pass_rate, accepted, feedback)

            self.optimization_memory.add_iteration(
                IterationSummary(
                    iteration=iteration,
                    prompt_change=prompt_change_desc,
                    target_issues=target_issues,
                    accepted=accepted,
                    training_before=previous_pass_rate,
                    training_after=new_pass_rate if accepted else previous_pass_rate,
                    outcome=outcome,
                )
            )

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
        training_cases: list[EvalCase],
        test_cases: list[EvalCase],
        starting_iteration: int,
        training_pass_rate: float,
    ) -> OptimizationResult:
        """Phase 2: Optimize on test set with descriptive feedback.

        Args:
            current_prompt: The prompt that passed training
            training_cases: Training evaluation cases (for validation)
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

        # Don't record initial test evaluation as a separate iteration
        # The first refinement iteration will start from starting_iteration

        test_iterations = 0
        iteration = starting_iteration

        # Track training regression from previous iteration
        previous_training_regression = None  # Will store (test_feedback, training_failures)

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

            # Compute current test stats from current results
            current_test_pass_rate, current_test_passed, current_test_total = self.test_runner.compute_pass_rate(results)

            # Generate feedback based on whether we had training regression last iteration
            if previous_training_regression is not None:
                # Use combined feedback (test patterns + training constraints)
                test_feedback, training_failures = previous_training_regression
                training_failed_count = len(training_failures)
                training_total_count = len(training_cases)
                training_regression_rate = 1.0 - (training_failed_count / training_total_count)

                feedback = CombinedFeedback(
                    test_pass_rate=current_test_pass_rate,
                    test_passed=current_test_passed,
                    test_total=current_test_total,
                    error_patterns=test_feedback.error_patterns,
                    training_pass_rate=training_regression_rate,
                    training_passed=training_total_count - training_failed_count,
                    training_total=training_total_count,
                    training_failures=training_failures,
                )

                print(f"\033[94mIteration {iteration}: Refining with training constraints...\033[0m")
                print(f"  Test pass rate: {current_test_passed}/{current_test_total} ({current_test_pass_rate:.1%})")
                print(f"  Training constraints: {len(training_failures)} cases to preserve")
                if test_feedback.error_patterns:
                    print(f"  Error patterns: {len(test_feedback.error_patterns)}")
                    for pattern in test_feedback.error_patterns:
                        print(f"   - {pattern.error_type.value}: {pattern.pattern_observed}")
                        print(f"     Example: {pattern.example_pattern}")
                        print(f"     Root cause: {pattern.root_cause}")
                        print(f"     Fix: {pattern.recommended_fix}")

                # Reset training regression tracking after using it
                previous_training_regression = None
            else:
                # Normal test feedback (patterns only, no specifics)
                feedback = self.feedback_analyzer.analyze_test_results(results)

                print(f"\033[94mIteration {iteration}: Refining based on test patterns...\033[0m")
                print(f"  Test pass rate: {current_test_passed}/{current_test_total} ({current_test_pass_rate:.1%})")
                print(f"  Error patterns: {len(feedback.error_patterns)}")
                for pattern in feedback.error_patterns:
                    print(f"   - {pattern.error_type.value}: {pattern.pattern_observed}")
                    print(f"     Example: {pattern.example_pattern}")
                    print(f"     Root cause: {pattern.root_cause}")
                    print(f"     Fix: {pattern.recommended_fix}")

            # Save current state before refinement
            previous_results = results
            previous_prompt = current_prompt
            previous_test_pass_rate = test_pass_rate
            previous_training_pass_rate = training_pass_rate

            # Generate multiple candidate prompts
            optimization_context = self.optimization_memory.get_context_string()
            num_candidates = self.config.optimizer.candidates_per_iteration

            # Determine which method to use
            if isinstance(feedback, CombinedFeedback):
                method_name = 'combined'
            else:
                method_name = 'test'

            print(f"  Generating {num_candidates} candidate prompts...")
            candidates = self.meta_optimizer.generate_multiple_candidates(
                base_method=method_name,
                num_candidates=num_candidates,
                current_prompt=current_prompt,
                feedback=feedback,
                optimization_context=optimization_context,
            )

            # Evaluate all candidates - prioritize those that maintain training & improve test
            best_candidate = None
            best_test_results = None
            best_test_pass_rate = previous_test_pass_rate
            best_training_pass_rate = previous_training_pass_rate

            had_training_regression = False
            training_regression_info = None

            for idx, candidate_prompt in enumerate(candidates, 1):
                print(f"  Evaluating candidate {idx}/{num_candidates}...")
                if self.verbose:
                    print(f"    Candidate {idx} prompt:\n{candidate_prompt}\n")

                # First, check training regression
                print(f"    Re-validating against training set...")
                training_results = self.test_runner.run_eval(candidate_prompt, training_cases)
                candidate_training_pass_rate, train_passed, train_total = self.test_runner.compute_pass_rate(training_results)

                if candidate_training_pass_rate < previous_training_pass_rate:
                    # Training regression - skip this candidate
                    print(f"    → Training: {previous_training_pass_rate:.1%} → {candidate_training_pass_rate:.1%} \033[91m✗ Regressed\033[0m")
                    had_training_regression = True

                    # Store for potential combined feedback next iteration
                    if training_regression_info is None:
                        training_failure_analysis = self.feedback_analyzer.analyze_training_results(training_results)
                        test_feedback_for_next = self.feedback_analyzer.analyze_test_results(results)
                        training_regression_info = (test_feedback_for_next, training_failure_analysis.failures)

                    continue  # Skip to next candidate

                # Training OK - evaluate on test set
                candidate_test_results = self.test_runner.run_eval(candidate_prompt, test_cases)
                candidate_test_pass_rate, test_passed, test_total = self.test_runner.compute_pass_rate(candidate_test_results)
                print(f"    → Training: {candidate_training_pass_rate:.1%}, Test: {test_passed}/{test_total} ({candidate_test_pass_rate:.1%})")

                # Track best candidate (use >= to allow lateral moves, matching acceptance logic)
                if candidate_test_pass_rate >= best_test_pass_rate:
                    best_test_pass_rate = candidate_test_pass_rate
                    best_candidate = candidate_prompt
                    best_test_results = candidate_test_results
                    best_training_pass_rate = candidate_training_pass_rate

            # Use the best candidate if we found one
            total_test_cases = len(test_cases)
            accepted = False
            new_test_pass_rate = previous_test_pass_rate
            new_training_pass_rate = previous_training_pass_rate

            if best_candidate is not None:
                # We found a candidate that maintains training
                candidate_prompt = best_candidate
                candidate_results = best_test_results
                new_test_pass_rate = best_test_pass_rate
                new_training_pass_rate = best_training_pass_rate

                # Validate: only accept if improvement or maintaining performance on test
                if self.prompt_history.should_accept(new_test_pass_rate, previous_test_pass_rate, phase="test"):
                    # Accept: update to new prompt and results
                    accepted = True
                    results = candidate_results
                    current_prompt = candidate_prompt
                    test_pass_rate = new_test_pass_rate
                    training_pass_rate = new_training_pass_rate
                    # Clear any pending training regression since we accepted a good prompt
                    previous_training_regression = None
                    print(f"  \033[92m✓ Accepted: test {previous_test_pass_rate:.1%} → {new_test_pass_rate:.1%}, training {previous_training_pass_rate:.1%} → {training_pass_rate:.1%}\033[0m")
                else:
                    # Test regressed - reject and restore previous state
                    results = previous_results  # Restore results to match current_prompt
                    current_prompt = previous_prompt
                    test_pass_rate = previous_test_pass_rate
                    new_training_pass_rate = previous_training_pass_rate  # Restore for summary
                    print(f"  \033[91m✗ Rejected (test regression): {previous_test_pass_rate:.1%} → {new_test_pass_rate:.1%}\033[0m")
                    print(f"  Keeping previous prompt")
            else:
                # No candidate improved test while maintaining training - reject all
                results = previous_results
                current_prompt = previous_prompt
                test_pass_rate = previous_test_pass_rate
                training_pass_rate = previous_training_pass_rate

                # Determine why candidates were rejected
                if had_training_regression:
                    print(f"  \033[91m✗ No candidate improved test score while maintaining training\033[0m")
                    print(f"  Keeping previous prompt")

                    # Store combined feedback for next iteration if any had training regression
                    if training_regression_info is not None:
                        previous_training_regression = training_regression_info
                        print(f"  Next iteration will refine with training constraints ({len(training_regression_info[1])} cases)")
                else:
                    # All maintained training but none improved test
                    print(f"  \033[91m✗ No candidate improved test score (all maintained training)\033[0m")
                    print(f"  Keeping previous prompt")

            print(f"  Current test pass rate: {int(test_pass_rate * total_test_cases)}/{total_test_cases} ({test_pass_rate:.1%})\n")

            # Record iteration summary in memory
            target_issues = self._extract_target_issues(feedback)
            prompt_change_desc = self._describe_prompt_change(feedback, phase="test")
            outcome = self._describe_test_outcome(
                previous_training_pass_rate, new_training_pass_rate,
                previous_test_pass_rate, new_test_pass_rate,
                accepted, feedback
            )

            self.optimization_memory.add_iteration(
                IterationSummary(
                    iteration=iteration,
                    prompt_change=prompt_change_desc,
                    target_issues=target_issues,
                    accepted=accepted,
                    training_before=previous_training_pass_rate,
                    training_after=training_pass_rate if accepted else previous_training_pass_rate,
                    test_before=previous_test_pass_rate,
                    test_after=test_pass_rate if accepted else previous_test_pass_rate,
                    outcome=outcome,
                )
            )

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

    def _extract_target_issues(self, feedback) -> list[str]:
        """Extract what issues this iteration is targeting.

        Args:
            feedback: Feedback object (DetailedFeedback or DescriptiveFeedback)

        Returns:
            List of issue descriptions
        """
        issues = []

        # Handle DetailedFeedback (training phase)
        if hasattr(feedback, 'failures'):
            # Count failures by error category
            error_counts = {}
            for failure in feedback.failures:
                category = failure.error_category.value
                error_counts[category] = error_counts.get(category, 0) + 1

            for category, count in error_counts.items():
                issues.append(f"{count} {category} error{'s' if count > 1 else ''}")

        # Handle DescriptiveFeedback (test phase)
        elif hasattr(feedback, 'error_patterns'):
            for pattern in feedback.error_patterns:
                issues.append(f"{pattern.count} {pattern.error_type.value} pattern{'s' if pattern.count > 1 else ''}")

        return issues

    def _describe_prompt_change(self, feedback, phase: str) -> str:
        """Generate description of what change was attempted.

        Args:
            feedback: Feedback object
            phase: "training" or "test"

        Returns:
            Description of the attempted change
        """
        # Extract main error types
        if hasattr(feedback, 'failures') and feedback.failures:
            main_errors = set(f.error_category.value for f in feedback.failures[:3])
            error_desc = ", ".join(main_errors)
            return f"Addressed {error_desc} errors from training feedback"

        elif hasattr(feedback, 'error_patterns') and feedback.error_patterns:
            main_patterns = [p.error_type.value for p in feedback.error_patterns[:2]]
            pattern_desc = ", ".join(main_patterns)
            return f"Refined prompt to handle {pattern_desc} patterns from test set"

        return "Attempted prompt refinement"

    def _describe_outcome(self, before: float, after: float, accepted: bool, feedback) -> str:
        """Describe the outcome of a training iteration.

        Args:
            before: Pass rate before
            after: Pass rate after
            accepted: Whether change was accepted
            feedback: Feedback object

        Returns:
            Outcome description
        """
        if not accepted:
            return f"Introduced regressions, broke previously working cases"

        # Calculate improvement
        improvement = after - before
        if improvement > 0:
            # Count failures fixed
            failures_before = int((1 - before) * feedback.total)
            failures_after = int((1 - after) * feedback.total)
            fixed_count = failures_before - failures_after
            return f"Fixed {fixed_count} failure{'s' if fixed_count != 1 else ''}, improved by {improvement:.1%}"
        else:
            return "Maintained performance"

    def _describe_test_outcome(
        self,
        train_before: float, train_after: float,
        test_before: float, test_after: float,
        accepted: bool, feedback
    ) -> str:
        """Describe the outcome of a test iteration.

        Args:
            train_before: Training pass rate before
            train_after: Training pass rate after
            test_before: Test pass rate before
            test_after: Test pass rate after
            accepted: Whether change was accepted
            feedback: Feedback object

        Returns:
            Outcome description
        """
        if train_after < train_before:
            return f"Regressed on training set, broke existing cases"

        if not accepted:
            return f"Regressed on test set, broke generalization"

        test_improvement = test_after - test_before
        if test_improvement > 0:
            return f"Improved test performance by {test_improvement:.1%}, maintained training"
        else:
            return "Maintained both training and test performance"
