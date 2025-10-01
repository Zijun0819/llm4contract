import logging
from typing import Optional
from .base import BaseClient
import os

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = 'anthropic'

logger = logging.getLogger(__name__)


class ClaudeClient(BaseClient):
    ClientClass = Anthropic

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

        # Initialize the Anthropic client
        # Uses api_key if provided, otherwise looks for ANTHROPIC_API_KEY env variable
        self.client = self.ClientClass(api_key=api_key)

        # Set default max tokens if not specified in model string
        self.max_tokens = 4096  # Default max tokens for Claude

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        """
        Convert messages to Claude format and generate responses
        """
        # Convert OpenAI-style messages to Claude format
        system_prompt, claude_messages = self._convert_messages_to_claude_format(messages)

        # Claude doesn't support n > 1 directly, so we'll make multiple calls if needed
        choices = []
        for i in range(n):
            try:
                # Create message with Claude API
                response = self.client.messages.create(
                    model=self.model,
                    messages=claude_messages,
                    system=system_prompt if system_prompt else None,
                    temperature=temperature,
                    max_tokens=self.max_tokens
                )

                # Convert Claude response to OpenAI-style format
                choice = self._convert_claude_response_to_choice(response)
                choices.append(choice)

            except Exception as e:
                logger.error(f"Error generating response {i + 1}/{n}: {e}")
                # Add empty choice on error
                choice = type('Choice', (), {
                    'message': type('Message', (), {'content': ''})
                })
                choices.append(choice)

        return choices

    def _convert_messages_to_claude_format(self, messages: list[dict]):
        """
        Convert OpenAI-style messages to Claude format

        OpenAI format: [{"role": "system/user/assistant", "content": "..."}]
        Claude format:
        - System message is separate
        - Messages list with {"role": "user/assistant", "content": "..."}
        """
        system_prompt = ""
        claude_messages = []

        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'system':
                # Claude handles system messages separately
                if system_prompt:
                    system_prompt += "\n\n" + content
                else:
                    system_prompt = content
            elif role == 'user':
                claude_messages.append({
                    'role': 'user',
                    'content': content
                })
            elif role == 'assistant':
                claude_messages.append({
                    'role': 'assistant',
                    'content': content
                })

        # Claude requires at least one message
        if not claude_messages:
            if system_prompt:
                # If only system message exists, convert it to a user message
                claude_messages.append({
                    'role': 'user',
                    'content': system_prompt
                })
                system_prompt = ""
            else:
                # Add a default message if completely empty
                claude_messages.append({
                    'role': 'user',
                    'content': ''
                })

        # Claude requires first message to be from user
        if claude_messages[0]['role'] != 'user':
            # If first message is from assistant, add an empty user message
            claude_messages.insert(0, {'role': 'user', 'content': 'Continue.'})

        # Claude requires alternating user/assistant messages
        claude_messages = self._ensure_alternating_messages(claude_messages)

        return system_prompt, claude_messages

    def _ensure_alternating_messages(self, messages: list[dict]) -> list[dict]:
        """
        Ensure messages alternate between user and assistant roles
        Claude API requires this strict alternation
        """
        if not messages:
            return messages

        fixed_messages = [messages[0]]

        for i in range(1, len(messages)):
            # Check if we have consecutive messages from the same role
            if messages[i]['role'] == fixed_messages[-1]['role']:
                if messages[i]['role'] == 'user':
                    # Insert an assistant message
                    fixed_messages.append({
                        'role': 'assistant',
                        'content': 'I understand.'
                    })
                else:
                    # Insert a user message
                    fixed_messages.append({
                        'role': 'user',
                        'content': 'Continue.'
                    })
            fixed_messages.append(messages[i])

        # Ensure last message is from user (Claude requirement for some models)
        # Actually, Claude typically expects the last message to be from user
        # but will accept assistant as last if that's the conversation flow

        return fixed_messages

    def _convert_claude_response_to_choice(self, response):
        """
        Convert Claude response to OpenAI-style choice format for compatibility
        """
        try:
            # Extract text content from Claude response
            content = response.content[0].text if response.content else ""

            # Create a mock choice object that mimics OpenAI's structure
            choice = type('Choice', (), {
                'message': type('Message', (), {
                    'content': content
                })
            })

        except Exception as e:
            logger.error(f"Error converting Claude response: {e}")
            # Return empty choice on error
            choice = type('Choice', (), {
                'message': type('Message', (), {'content': ''})
            })

        return choice
