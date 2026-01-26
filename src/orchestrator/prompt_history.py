"""Prompt version history tracking for adaptive optimization."""

from dataclasses import dataclass


@dataclass
class PromptVersion:
    """A versioned prompt with its performance metrics."""

    iteration: int
    prompt: str
    training_score: float
    test_score: float | None = None


class PromptHistory:
    """Tracks prompt versions and manages rollback decisions."""

    def __init__(self):
        """Initialize empty history."""
        self.versions: list[PromptVersion] = []
        self.accepted_count = 0
        self.rejected_count = 0

    def add_version(
        self,
        iteration: int,
        prompt: str,
        training_score: float,
        test_score: float | None = None,
    ) -> None:
        """Add a new prompt version to history.

        Args:
            iteration: Iteration number
            prompt: The prompt text
            training_score: Training pass rate (0.0 to 1.0)
            test_score: Test pass rate (0.0 to 1.0), optional
        """
        version = PromptVersion(
            iteration=iteration,
            prompt=prompt,
            training_score=training_score,
            test_score=test_score,
        )
        self.versions.append(version)

    def should_accept(
        self,
        new_score: float,
        current_score: float,
        phase: str = "training",
    ) -> bool:
        """Determine if a new prompt should be accepted.

        Args:
            new_score: Score of the candidate prompt
            current_score: Score of the current prompt
            phase: "training" or "test" phase

        Returns:
            True if the new prompt should be accepted
        """
        # Training phase logic:
        # - If current score is very low (< 10%), require strict improvement to avoid 0% → 0% loops
        # - If current score is decent (>= 10%), allow lateral moves to explore different approaches
        # - This helps break through plateaus at high percentages (e.g., 98% → 98% is OK)
        if phase == "training":
            if current_score < 0.1:
                # Very low score - require improvement to avoid getting stuck at 0%
                accept = new_score > current_score
            else:
                # Decent score - allow lateral moves to explore
                accept = new_score >= current_score
        else:
            # Test phase: allow equal performance (trying different approaches)
            accept = new_score >= current_score

        if accept:
            self.accepted_count += 1
        else:
            self.rejected_count += 1

        return accept

    def get_best_version(self, phase: str = "training") -> PromptVersion | None:
        """Get the best performing prompt version.

        Args:
            phase: "training" or "test" - which score to optimize for

        Returns:
            Best prompt version, or None if history is empty
        """
        if not self.versions:
            return None

        if phase == "training":
            return max(self.versions, key=lambda v: v.training_score)
        else:
            # For test phase, prioritize test score, fallback to training score
            test_versions = [v for v in self.versions if v.test_score is not None]
            if test_versions:
                return max(test_versions, key=lambda v: v.test_score or 0)
            return max(self.versions, key=lambda v: v.training_score)

    def get_current_version(self) -> PromptVersion | None:
        """Get the most recent prompt version.

        Returns:
            Most recent version, or None if history is empty
        """
        return self.versions[-1] if self.versions else None

    def get_acceptance_rate(self) -> float:
        """Calculate acceptance rate of refinements.

        Returns:
            Acceptance rate (0.0 to 1.0)
        """
        total = self.accepted_count + self.rejected_count
        return self.accepted_count / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, int | float]:
        """Get history statistics.

        Returns:
            Dict with acceptance/rejection counts and rate
        """
        return {
            "total_versions": len(self.versions),
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "acceptance_rate": self.get_acceptance_rate(),
        }
