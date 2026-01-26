"""Meta-optimizer that generates and refines prompts using an LLM."""

from src.providers.base import LLMProvider
from .models import DescriptiveFeedback, DetailedFeedback, EvalCase
from .prompts import render_initial_prompt, render_test_refinement, render_training_refinement


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
    ) -> str:
        """Refine prompt based on training set feedback (full details).

        Args:
            current_prompt: The current prompt to refine
            feedback: Detailed feedback with all failure information

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
    ) -> str:
        """Refine prompt based on test set feedback (descriptive patterns only).

        Args:
            current_prompt: The current prompt to refine
            feedback: Descriptive feedback without specific test cases

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

    def get_total_tokens_used(self) -> int:
        """Get total tokens used by this optimizer.

        Returns:
            Total token count
        """
        return self.tokens_used
