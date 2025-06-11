import logging
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
        # 将numpy数组转换回PIL图像
        image = Image.fromarray(image[:, :, :3], mode='RGB')

        # 将图像保存到字节流中
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        # 将字节流编码为Base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")


class GeminiVLM:
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-2.0-flash", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        self.client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=os.environ.get("GEMINI_BASE_URL")
        )

        self.system_instruction = system_instruction

        self.spend = 0
        if '1.5-flash' in self.name:
            self.cost_per_input_token = 0.075 / 1_000_000
            self.cost_per_output_token = 0.3 / 1_000_000
        elif '1.5-pro' in self.name:
            self.cost_per_input_token = 1.25 / 1_000_000
            self.cost_per_output_token = 5 / 1_000_000
        else:
            self.cost_per_input_token = 0.1 / 1_000_000
            self.cost_per_output_token = 0.4 / 1_000_000

    def call_chat(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
                           response.usage.completion_tokens * self.cost_per_output_token)
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def call(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens * self.cost_per_input_token +
                           response.usage.completion_tokens * self.cost_per_output_token)
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend

class QwenVLM:
    """
    A specific implementation of a VLM using the Qwen API for image and text inference.
    """

    def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct", system_instruction=None):
        """
        Initialize the Qwen model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        self.client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url=os.environ.get("GEMINI_BASE_URL")
        )

        self.system_instruction = system_instruction

        self.spend = 0

    def call_chat(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)

        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def call(self, image: list[np.array], text_prompt: str):
        base64_image = encode_image(image[0])
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False  # 是否开启流式输出
            )
            self.spend += (response.usage.prompt_tokens + response.usage.completion_tokens)
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend



