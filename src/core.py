from __future__ import annotations

__version__ = "0.1.0"


class Explainer:
    """Minimal example explainer used by notebooks/tests.

    This can be extended to support your thesis logic.
    """

    def explain(self, text: str) -> str:
        """Return a simple explanation for the provided text.

        Args:
            text: The input string to explain.

        Returns:
            A short string explanation.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        return f"Explanation for: {text.strip()}"

