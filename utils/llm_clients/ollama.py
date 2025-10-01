import logging
from typing import Optional
from .base import BaseClient

try:
    from together import Together
except ImportError:
    Together = 'together'

logger = logging.getLogger(__name__)


class OllamaClient(BaseClient):
    ClientClass = Together

    def __init__(
            self,
            model: str,
            temperature: float = 1.0,
            api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature)

        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)

        self.client = self.ClientClass()

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, n=n, stream=False, max_tokens=120000,
        )
        return response.choices
