"""Maintains optimization context across iterations."""

from collections import deque
from dataclasses import dataclass


@dataclass
class IterationSummary:
    """Detailed summary of what happened in an iteration."""

    iteration: int
    prompt_change: str  # Concrete description of what was modified
    target_issues: list[str]  # What failures it aimed to fix
    accepted: bool
    training_before: float
    training_after: float
    test_before: float | None = None
    test_after: float | None = None
    outcome: str = ""  # Concrete result description


class OptimizationMemory:
    """Maintains context across iterations for the meta-optimizer."""

    def __init__(self, max_recent_history: int = 3):
        """Initialize optimization memory.

        Args:
            max_recent_history: Number of recent iterations to keep in context
        """
        self.lessons_learned: list[str] = []
        self.recent_history: deque[IterationSummary] = deque(maxlen=max_recent_history)

    def add_iteration(self, summary: IterationSummary):
        """Add an iteration to the history.

        Args:
            summary: Summary of the iteration
        """
        self.recent_history.append(summary)

        # Extract lesson and add to accumulated wisdom
        if summary.accepted:
            lesson = f"✓ {summary.prompt_change} improved performance"
        else:
            lesson = f"✗ {summary.prompt_change} caused regression"

        self.lessons_learned.append(lesson)

    def get_context_string(self) -> str:
        """Generate formatted context string for meta-optimizer.

        Returns:
            Formatted string with lessons and recent history
        """
        if not self.recent_history and not self.lessons_learned:
            return ""

        lines = []
        lines.append("Optimization Context:")
        lines.append("━" * 70)

        # Lessons learned section
        if self.lessons_learned:
            lines.append("Accumulated Lessons:")
            for lesson in self.lessons_learned[-5:]:  # Last 5 lessons
                lines.append(f"• {lesson}")
            lines.append("")

        # Recent history section
        if self.recent_history:
            lines.append("Recent Iteration History:")
            lines.append("━" * 70)

            for summary in self.recent_history:
                status = "✓ ACCEPTED" if summary.accepted else "✗ REJECTED"
                lines.append(f"Iteration {summary.iteration}: {status}")
                lines.append(f"  Change Made: {summary.prompt_change}")
                lines.append("")

                if summary.target_issues:
                    lines.append(f"  Target Issues: {', '.join(summary.target_issues)}")
                    lines.append("")

                # Training result
                training_change = f"{summary.training_before:.1%} → {summary.training_after:.1%}"
                result_line = f"  Result: Training {training_change}"

                # Test result if in test phase
                if summary.test_before is not None and summary.test_after is not None:
                    test_change = f"{summary.test_before:.1%} → {summary.test_after:.1%}"
                    result_line += f", Test {test_change}"

                if not summary.accepted:
                    result_line += " (REGRESSION)"

                lines.append(result_line)

                if summary.outcome:
                    lines.append(f"          {summary.outcome}")

                lines.append("")
                lines.append("━" * 70)

        return "\n".join(lines)
