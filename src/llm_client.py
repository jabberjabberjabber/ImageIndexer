"""HTTP client for the vision LLM.
"""
import requests

def _json_schema(name, schema):
    """Wrap a JSON schema.
    """
    
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema},
    }


_KEYWORDS_SCHEMA = _json_schema("keywords", {
    "type": "object",
    "properties": {
        "Keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["Keywords"],
})

_CAPTION_AND_KEYWORDS_SCHEMA = _json_schema("caption_and_keywords", {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["Description", "Keywords"],
})

_SCHEMAS = {
    "keywords": _KEYWORDS_SCHEMA,
    "caption_and_keywords": _CAPTION_AND_KEYWORDS_SCHEMA,
}

# Prefill and grammar cannot be combined.

_PREFILLS = {
    "caption": "A ",
    "keywords": '{"Keywords": ["',
    "caption_and_keywords": '{"Description": "A ',
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

        self._use_prefill = getattr(config, "use_prefill", False)
        self._prefills = dict(_PREFILLS)
        self._prefills.update(getattr(config, "prefills", None) or {})
        self._model_name = None

    def _prefill_for(self, task, grammar_applied):
        """Return the prefill string.
        """
        
        if not self._use_prefill or grammar_applied:
            return ""
        return self._prefills.get(task) or ""

    def model_name(self):
        """Identify the loaded model, for tagging results by their source.
        """
        
        if self._model_name is None:
            self._model_name = "unknown"
            try:
                response = self.session.get(f"{self.api_url}/v1/models", timeout=10)
                response.raise_for_status()
                data = response.json().get("data") or []
                if data and data[0].get("id"):
                    self._model_name = str(data[0]["id"])
            except Exception:
                pass
        return self._model_name

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

        grammar_applied = self.config.use_json_grammar and task in _SCHEMAS
        if grammar_applied:
            payload["response_format"] = _SCHEMAS[task]

        prefill = self._prefill_for(task, grammar_applied)
        if prefill:
            payload["messages"].append({"role": "assistant", "content": prefill})
            payload["continue_assistant_turn"] = True

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
                
                # The reply is only the continuation
                return prefill + content if prefill else content

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
