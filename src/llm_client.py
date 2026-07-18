"""HTTP client for the vision LLM.
"""
import requests

_KEYWORDS_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "Keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["Keywords"],
    },
}

_CAPTION_AND_KEYWORDS_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "Description": {"type": "string"},
            "Keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["Description", "Keywords"],
    },
}

_SCHEMAS = {
    "keywords": _KEYWORDS_SCHEMA,
    "caption_and_keywords": _CAPTION_AND_KEYWORDS_SCHEMA,
}


class LLMProcessor:
    def __init__(self, config):
        self.config = config
        self.api_url = config.api_url
        self.endpoint = f"{config.api_url}/v1/chat/completions"

        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        if config.api_password:
            self.session.headers["Authorization"] = f"Bearer {config.api_password}"

        self._base_payload = {
            "max_tokens": config.gen_count,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "min_p": config.min_p,
            "rep_pen": config.rep_pen,
            "use_default_badwordsids": config.use_default_badwordsids,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        self._instructions = {
            "caption": config.caption_instruction,
            "keywords": config.tag_instruction,
            "caption_and_keywords": config.instruction,
        }

    def describe_content(self, task="", processed_image=None):
        """Send one image + instruction to the API; return the raw text reply."""
        if not processed_image:
            print("No image to describe.")
            return None

        instruction = self._instructions.get(task)
        if instruction is None:
            print(f"invalid task: {task}")
            return None

        payload = dict(self._base_payload)
        payload["messages"] = [
            {"role": "system", "content": self.config.system_instruction},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{processed_image}"
                        },
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        if self.config.use_json_grammar and task in _SCHEMAS:
            payload["response_format"] = _SCHEMAS[task]

        try:
            response = self.session.post(self.endpoint, json=payload, timeout=600)
            response.raise_for_status()
            response_json = response.json()

            choices = response_json.get("choices")
            if choices:
                choice = choices[0]
                if "message" in choice:
                    content = choice["message"]["content"]
                else:
                    content = choice.get("text", "")
                print(f"  Received response from API ({len(content)} chars)")
                return content

            print("  Warning: API response missing expected data")
            return None

        except requests.exceptions.ConnectionError:
            print(f"API Connection Error: Cannot connect to {self.api_url}")
            print("  Make sure the LLM server is running and accessible")
        except requests.exceptions.Timeout:
            print(f"API Timeout Error: Request to {self.api_url} timed out")
        except requests.exceptions.HTTPError as e:
            print(f"API HTTP Error: {e.response.status_code} - {str(e)}")
            if hasattr(e.response, "text"):
                print(f"  Response: {e.response.text[:200]}")
        except Exception as e:
            print(f"API Error: {type(e).__name__} - {str(e)}")
        return None
