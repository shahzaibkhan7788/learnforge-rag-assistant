"""LearnForge support assistant application package.

The package deliberately keeps its default runtime dependency-free. Optional
vector databases and LLM providers are loaded only when explicitly configured.
"""

from .main import SupportAssistant

__all__ = ["SupportAssistant"]
