import logging
import time
import threading
from abc import ABC, abstractmethod
from openai import OpenAI
import base64
import numpy as np
from PIL import Image
import io
import os

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def encode_image(image):
    try:
        image = Image.fromarray(image[:, :, :3], mode='RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")


class BaseVLM(ABC):
    """Abstract base class for VLM implementations."""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0

    def __init__(self, model: str, system_instruction=None):
        self.name = model
        self.system_instruction = system_instruction
        self._spend = 0.0
        self._spend_lock = threading.Lock()
        self.client = self._build_client()

    @abstractmethod
    def _build_client(self) -> OpenAI:
        """Construct and return the OpenAI-compatible client."""

    @abstractmethod
    def _add_spend(self, response) -> None:
        """Update spend based on the API response."""

    @abstractmethod
    def _image_url_payload(self, base64_image: str) -> dict:
        """Return the image_url content block appropriate for this backend."""

    def _make_messages(self, base64_image: str, text_prompt: str, with_system: bool) -> list:
        image_block = {
            "type": "image_url",
            "image_url": self._image_url_payload(base64_image),
        }
        user_content = [{"type": "text", "text": text_prompt}, image_block]
        messages = []
        if with_system and self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _call(self, messages: list) -> str:
        last_err = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=500,
                    temperature=0,
                    top_p=1,
                    stream=False,
                )
                self._add_spend(response)
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    logging.warning(f"{self.__class__.__name__} API error (attempt {attempt + 1}): {e}. Retrying in {delay}s.")
                    time.sleep(delay)
        print(f"{self.__class__.__name__} API ERROR: {last_err}")
        return f"{self.__class__.__name__} API ERROR"

    def call_chat(self, image: list, text_prompt: str) -> str:
        base64_image = encode_image(image[0])
        return self._call(self._make_messages(base64_image, text_prompt, with_system=True))

    def call(self, image: list, text_prompt: str) -> str:
        base64_image = encode_image(image[0])
        return self._call(self._make_messages(base64_image, text_prompt, with_system=False))

    def reset(self):
        """Reset the context state of the VLM agent."""
        pass

    def get_spend(self) -> float:
        with self._spend_lock:
            return self._spend


class GeminiVLM(BaseVLM):
    """VLM implementation backed by the Gemini API."""

    def __init__(self, model="gemini-2.0-flash", system_instruction=None):
        if '1.5-flash' in model:
            self._cost_per_input = 0.075 / 1_000_000
            self._cost_per_output = 0.3 / 1_000_000
        elif '1.5-pro' in model:
            self._cost_per_input = 1.25 / 1_000_000
            self._cost_per_output = 5 / 1_000_000
        else:
            self._cost_per_input = 0.1 / 1_000_000
            self._cost_per_output = 0.4 / 1_000_000
        super().__init__(model, system_instruction)

    def _build_client(self) -> OpenAI:
        return OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=os.environ.get("GEMINI_BASE_URL"),
        )

    def _add_spend(self, response) -> None:
        cost = (response.usage.prompt_tokens * self._cost_per_input +
                response.usage.completion_tokens * self._cost_per_output)
        with self._spend_lock:
            self._spend += cost

    def _image_url_payload(self, base64_image: str) -> str:
        return f"data:image/png;base64,{base64_image}"


class QwenVLM(BaseVLM):
    """VLM implementation backed by a Qwen2.5-VL-compatible API."""

    def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct", system_instruction=None):
        super().__init__(model, system_instruction)

    def _build_client(self) -> OpenAI:
        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GEMINI_API_KEY")
        base_url = os.environ.get("QWEN_BASE_URL") or os.environ.get("GEMINI_BASE_URL")
        return OpenAI(api_key=api_key, base_url=base_url)

    def _add_spend(self, response) -> None:
        tokens = response.usage.prompt_tokens + response.usage.completion_tokens
        with self._spend_lock:
            self._spend += tokens

    def _image_url_payload(self, base64_image: str) -> dict:
        return {"url": f"data:image/png;base64,{base64_image}"}
