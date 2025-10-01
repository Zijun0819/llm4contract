import logging
from typing import Optional
from .base import BaseClient
import os

try:
    from google import genai
except ImportError:
    genai = 'google-genai'

logger = logging.getLogger(__name__)


class GeminiClient(BaseClient):
    ClientClass = genai

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

        # Set API key in environment variable if provided
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key

        # Initialize the client (gets API key from GEMINI_API_KEY env variable)
        self.client = self.ClientClass.Client()

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        """
        Convert messages to Gemini format and generate responses
        """
        # Convert OpenAI-style messages to Gemini format
        contents = self._convert_messages_to_gemini_format(messages)

        # Configure generation settings
        config = {
            'temperature': temperature,
            'candidate_count': n,  # Number of responses to generate
        }

        # Generate response using the new API
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )

        # Convert Gemini response to OpenAI-style format for compatibility
        return self._convert_gemini_response_to_choices(response, n)

    def _convert_messages_to_gemini_format(self, messages: list[dict]):
        """
        Convert OpenAI-style messages to Gemini format

        OpenAI format: [{"role": "system/user/assistant", "content": "..."}]
        Gemini format: Can be a string for simple queries or list of dicts for conversations
        """
        # Handle system messages by combining with first user message
        system_prompt = ""
        converted_messages = []

        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'system':
                # Store system message to prepend to first user message
                system_prompt = content
            elif role == 'user':
                # Combine system prompt with user message if exists
                if system_prompt:
                    content = f"{system_prompt}\n\n{content}"
                    system_prompt = ""  # Reset after using

                # For multi-turn conversations, build proper format
                converted_messages.append({
                    'role': 'user',
                    'parts': [{'text': content}]
                })
            elif role == 'assistant':
                converted_messages.append({
                    'role': 'model',
                    'parts': [{'text': content}]
                })

        # If only system message exists, treat it as user message
        if system_prompt and not converted_messages:
            converted_messages.append({
                'role': 'user',
                'parts': [{'text': system_prompt}]
            })

        # For single user message, can return just the string
        if len(converted_messages) == 1 and converted_messages[0]['role'] == 'user':
            return converted_messages[0]['parts'][0]['text']

        # For multi-turn conversations, return the full structure
        return converted_messages

    def _convert_gemini_response_to_choices(self, response, n: int):
        """
        Convert Gemini response to OpenAI-style choices format for compatibility
        """
        choices = []

        # The response object from the new API
        try:
            # Handle the response text directly
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'candidates'):
                # Handle multiple candidates if available
                for i, candidate in enumerate(response.candidates[:n]):
                    text = candidate.content.parts[0].text if hasattr(candidate.content, 'parts') else str(candidate)
                    choice = type('Choice', (), {
                        'message': type('Message', (), {
                            'content': text
                        })
                    })
                    choices.append(choice)
                return choices
            else:
                content = str(response)

            # Create a mock choice object that mimics OpenAI's structure
            choice = type('Choice', (), {
                'message': type('Message', (), {
                    'content': content
                })
            })
            choices.append(choice)

        except Exception as e:
            logger.error(f"Error converting Gemini response: {e}")
            # Return empty choice on error
            choice = type('Choice', (), {
                'message': type('Message', (), {'content': ''})
            })
            choices.append(choice)

        # If we need more responses than we got, duplicate the last one
        # (Note: Gemini might not always return n candidates even if requested)
        while len(choices) < n:
            if choices:
                choices.append(choices[-1])
            else:
                choices.append(type('Choice', (), {
                    'message': type('Message', (), {'content': ''})
                }))

        return choices
