"""Meta-optimizer that generates and refines prompts using an LLM."""

from src.providers.base import LLMProvider
from .models import CombinedFeedback, DescriptiveFeedback, DetailedFeedback, EvalCase
from .prompts import (
    render_combined_refinement,
    render_initial_prompt,
    render_test_refinement,
    render_training_refinement,
)


class MetaOptimizer:
    """Uses an LLM to generate and iteratively refine prompts."""

    def __init__(self, provider: LLMProvider):
        """Initialize the meta-optimizer.

        Args:
            provider: LLM provider to use for optimization (e.g., Opus)
        """
        self.provider = provider
        self.tokens_used = 0

    def generate_initial_prompt(
        self,
        task_description: str,
        training_examples: list[EvalCase],
        max_examples: int = 10,
    ) -> str:
        """Generate an initial zero-shot prompt based on task and examples.

        Args:
            task_description: Description of the task to optimize for
            training_examples: Example input/output pairs
            max_examples: Maximum number of examples to show to the optimizer

        Returns:
            Initial prompt text
        """
        # Limit examples shown to optimizer to avoid huge context
        examples_to_show = training_examples[:max_examples]

        # Render the meta-prompt
        meta_prompt = render_initial_prompt(task_description, examples_to_show)

        # Generate using the LLM provider
        response = self.provider.generate(
            prompt=meta_prompt,
            temperature=1.0,  # Want creativity for initial generation
            max_tokens=2000,
        )

        # Track token usage
        self.tokens_used += self.provider.count_tokens(meta_prompt)
        self.tokens_used += self.provider.count_tokens(response)

        return response.strip()

    def refine_prompt_training(
        self,
        current_prompt: str,
        feedback: DetailedFeedback,
        optimization_context: str = "",
    ) -> str:
        """Refine prompt based on training set feedback (full details).

        Args:
            current_prompt: The current prompt to refine
            feedback: Detailed feedback with all failure information
            optimization_context: Context from previous iterations

        Returns:
            Refined prompt text
        """
        # Render the refinement meta-prompt
        meta_prompt = render_training_refinement(
            current_prompt=current_prompt,
            pass_rate=feedback.pass_rate,
            passed=feedback.passed,
            total=feedback.total,
            failures=feedback.failures,
            optimization_context=optimization_context,
        )

        # Generate refined prompt
        response = self.provider.generate(
            prompt=meta_prompt,
            temperature=0.7,  # Slightly creative but focused
            max_tokens=2000,
        )

        # Track token usage
        self.tokens_used += self.provider.count_tokens(meta_prompt)
        self.tokens_used += self.provider.count_tokens(response)

        return response.strip()

    def refine_prompt_test(
        self,
        current_prompt: str,
        feedback: DescriptiveFeedback,
        optimization_context: str = "",
    ) -> str:
        """Refine prompt based on test set feedback (descriptive patterns only).

        Args:
            current_prompt: The current prompt to refine
            feedback: Descriptive feedback without specific test cases
            optimization_context: Context from previous iterations

        Returns:
            Refined prompt text
        """
        # Render the test refinement meta-prompt
        meta_prompt = render_test_refinement(
            current_prompt=current_prompt,
            pass_rate=feedback.pass_rate,
            passed=feedback.passed,
            total=feedback.total,
            error_patterns=feedback.error_patterns,
            optimization_context=optimization_context,
        )

        # Generate refined prompt
        response = self.provider.generate(
            prompt=meta_prompt,
            temperature=0.7,  # Slightly creative but focused on generalization
            max_tokens=2000,
        )

        # Track token usage
        self.tokens_used += self.provider.count_tokens(meta_prompt)
        self.tokens_used += self.provider.count_tokens(response)

        return response.strip()

    def refine_prompt_combined(
        self,
        current_prompt: str,
        feedback: CombinedFeedback,
        optimization_context: str = "",
    ) -> str:
        """Refine prompt with both test patterns and training constraints.

        Used when previous refinement caused training regression.

        Args:
            current_prompt: The current prompt to refine
            feedback: Combined feedback with test patterns and training constraints
            optimization_context: Context from previous iterations

        Returns:
            Refined prompt text
        """
        # Render the combined refinement meta-prompt
        meta_prompt = render_combined_refinement(
            current_prompt=current_prompt,
            test_pass_rate=feedback.test_pass_rate,
            test_passed=feedback.test_passed,
            test_total=feedback.test_total,
            error_patterns=feedback.error_patterns,
            training_pass_rate=feedback.training_pass_rate,
            training_passed=feedback.training_passed,
            training_total=feedback.training_total,
            training_failures=feedback.training_failures,
            optimization_context=optimization_context,
        )

        # Generate refined prompt
        response = self.provider.generate(
            prompt=meta_prompt,
            temperature=0.7,  # Balanced creativity to handle constraints
            max_tokens=2000,
        )

        # Track token usage
        self.tokens_used += self.provider.count_tokens(meta_prompt)
        self.tokens_used += self.provider.count_tokens(response)

        return response.strip()

    def generate_multiple_candidates(
        self,
        base_method: str,
        num_candidates: int,
        **kwargs
    ) -> list[str]:
        """Generate multiple candidate prompts.

        Args:
            base_method: Which refinement method to use ('training', 'test', or 'combined')
            num_candidates: Number of candidates to generate
            **kwargs: Arguments to pass to the refinement method

        Returns:
            List of candidate prompts (may contain duplicates if temperature is low)
        """
        # Select the appropriate method
        method_map = {
            'training': self.refine_prompt_training,
            'test': self.refine_prompt_test,
            'combined': self.refine_prompt_combined,
        }

        if base_method not in method_map:
            raise ValueError(f"Unknown method: {base_method}")

        method = method_map[base_method]
        candidates = []

        # Generate multiple candidates - diversity comes from temperature > 0
        for i in range(num_candidates):
            candidate = method(**kwargs)
            candidates.append(candidate)

        return candidates

    def get_total_tokens_used(self) -> int:
        """Get total tokens used by this optimizer.

        Returns:
            Total token count
        """
        return self.tokens_used
