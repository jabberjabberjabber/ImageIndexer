import os, json, time, re, argparse, exiftool, threading, queue, calendar, io, uuid, requests
from json_repair import repair_json as rj
from datetime import timedelta
from .image_processor import ImageProcessor
from .llmii_utils import first_json, de_pluralize, AND_EXCEPTIONS
    
def split_on_internal_capital(word):
    """ Split a word if it contains a capital letter after the 4th position.
        Returns the original word if no split is needed, or the split 
        version if a capital is found.
        
        Examples:
            BlueSky -> Blue Sky
            microService -> micro Service
    """
    if len(word) <= 4:
        return word
    
    for i in range(4, len(word)):
        if word[i].isupper():
            return word[:i] + " " + word[i:]
            
    return word

def normalize_keyword(keyword, banned_words, config=None):
    """ Normalizes keywords according to specific rules:
        - Splits unhyphenated compound words on internal capitals
        - Max words determined by config (default 2) unless middle word is 'and'/'or' (then +1)
        - If split_and_entries enabled, remove and/or unless in exceptions list
        - Hyphens between alphanumeric chars count as two words
        - Cannot start with 3+ digits if no_digits_start is enabled
        - Each word must be 2+ chars if min_word_length enabled (unless it is x or u)
        - Removes all non-alphanumeric except spaces and valid hyphens
        - Checks against banned words if ban_prompt_words enabled
        - Makes singular if depluralize_keywords enabled
        - Returns lowercase result
    """   
    if config is None:
        class DefaultConfig:
            def __init__(self):
                self.normalize_keywords = True
                self.depluralize_keywords = True
                self.limit_word_count = True
                self.max_words_per_keyword = 2
                self.split_and_entries = True
                self.ban_prompt_words = True
                self.no_digits_start = True
                self.min_word_length = True
                self.latin_only = True
        
        config = DefaultConfig()
    
    if not config.normalize_keywords:
        return keyword.strip()
    
    if not isinstance(keyword, str):
        keyword = str(keyword)
    
    # Handle internal capitalization before lowercase conversion
    words = keyword.strip().split()
    split_words = []
    
    for word in words:
        split_words.extend(split_on_internal_capital(word).split())
    
    keyword = " ".join(split_words)
    
    # Convert to lowercase after handling capitals
    keyword = keyword.lower().strip()
    
    # Remove non-Latin characters if latin_only is enabled
    if config.latin_only:
        keyword = re.sub(r'[^\x00-\x7F]', '', keyword)
    
    # Remove all non-alphanumeric chars except spaces and hyphens
    keyword = re.sub(r'[^\w\s-]', '', keyword)
    
    # Replace multiple spaces/hyphens with single space/hyphen
    keyword = re.sub(r'\s+', ' ', keyword)
    keyword = re.sub(r'-+', '-', keyword)
    keyword = re.sub(r'_', ' ', keyword)
    
    # Check for banned words if enabled
    if config.ban_prompt_words and keyword in banned_words:
        return None
    
    # For validation, we'll track both original tokens and split words
    tokens = keyword.split()
    words = []
    
    # Validate and collect words for length checking
    for token in tokens:    
        
        # Handle hyphenated words
        if '-' in token:
            
            # Check if hyphen is between alphanumeric chars
            if not re.match(r'^[\w]+-[\w]+$', token):
                return None
           
            # Add hyphenated parts to words list for validation
            parts = token.split('-')
            words.extend(parts)
        
        else:
            words.append(token)
    
    # Validate word count if limit_word_count is enabled
    if config.limit_word_count:
        max_words = config.max_words_per_keyword
        if len(words) > max_words + 1:
            return None
        
    # Handle and/or splitting if enabled
    if config.split_and_entries and len(words) == 3 and words[1] in ['and', 'or']:
        if ' '.join(words) in AND_EXCEPTIONS:
            pass
        else:
            # Remove and/or and make singular if depluralize_keywords is enabled
            if config.depluralize_keywords:
                tokens = [de_pluralize(words[0]), de_pluralize(words[2])]
            else:
                tokens = [words[0], words[2]]
    
    # Word validation
    for word in words:
        
        # Check minimum length if enabled
        if config.min_word_length:
            if len(word) < 2 and word not in ['x', 'u']:
                return None
        
    # Check if starts with 3+ digits if enabled
    if config.no_digits_start and words and re.match(r'^\d{3,}', words[0]):
        return None
    
    # Make words singular if depluralize_keywords is enabled
    if config.depluralize_keywords:
        # Make solo words singular
        if len(words) == 1:
            tokens = [de_pluralize(words[0])]
        # If two or more words make the last word singular
        elif len(tokens) > 1:
            tokens[-1] = de_pluralize(tokens[-1])
    
    # Return the original tokens (preserving hyphens)
    return ' '.join(tokens)
    
def clean_string(data):
    """ Makes sure the string is clean for addition
        to the metadata.
    """
        
    if isinstance(data, dict):
        data = json.dumps(data)
    
    # Remove <think> content
    if isinstance(data, str):
        # Remove matched pairs first
        data = re.sub(r'<think>.*?</think>', '', data, flags=re.DOTALL)
        # Remove any remaining orphaned opening tags
        data = re.sub(r'<think>', '', data)
        # Remove any remaining orphaned closing tags
        data = re.sub(r'</think>', '', data)
        
        # Normalize
        data = re.sub(r"\n", "", data)
        data = re.sub(r'["""]', '"', data)
        data = re.sub(r"\\{2}", "", data)
        last_period = data.rfind('.')
        
        if last_period != -1:
            data = data[:last_period+1]
    else:
        return ""
        
    return data
    

def markdown_list_to_dict(text):
    """ Searches a string for a markdown formatted
        list, and if one is found, converts it to
        a dict.
    """
    list_pattern = r"(?:^\s*[-*+]|\d+\.)\s*(.+)$"
    list_items = re.findall(list_pattern, text, re.MULTILINE)

    if list_items:
        return {"Keywords": list_items}
    else:
        return None
        
def clean_json(data):
    """ LLMs like to return all sorts of garbage.
        Even when asked to give a structured output
        they will wrap text around it explaining why
        they chose certain things. This function
        will pull basically anything useful and turn it
        into a dict.

        Handles various formats including:
        - Direct dicts
        - List-wrapped dicts: [{"Description": ...}]
        - String JSON with markdown wrappers
        - Malformed JSON requiring repair
    """
    if data is None:
        return None

    if isinstance(data, dict):
        return data

    # Handle list-wrapped dicts (sometimes APIs return [{"Description": ...}])
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            return data[0]

    if isinstance(data, str):
        # Try direct JSON parsing first (works with JSON grammar)
        try:
            result = json.loads(data)
            # If result is a list with a dict, unwrap it
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return result[0]
            return result
        except:
            pass

        # Try to extract JSON markdown code
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, data, re.DOTALL)
        if match:
            data = match.group(1).strip()
            try:
                result = json.loads(data)
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    return result[0]
                return result
            except:
                pass

        # Fallback: Try with repair_json
        try:
            result = json.loads(rj(data))
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return result[0]
            return result
        except:
            pass

        # Fallback: first_json + repair_json
        try:
            result = json.loads(rj(first_json(data)))
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return result[0]
            return result
        except:
            pass

        # Nuclear option: wrap in brackets and repair
        try:
            result = json.loads(first_json(rj("{" + data + "}")))
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                result = result[0]
            if result.get("Keywords"):
                return result
        except:
            pass
    
        # Strangelove option
        try:
            return markdown_list_to_dict(data)
        except:
            pass

    return None

def clean_tags(data):
    """ Extract and combine all Keywords entries from LLM output.
        When EOS token is banned, the model may generate multiple
        JSON objects with Keywords arrays. This function finds all
        of them and combines them into a single Keywords list.

        Returns a dict with a single Keywords key containing all found keywords.
    """
    all_keywords = []

    if data is None:
        return None

    if isinstance(data, dict):
        # Single dict - extract Keywords if present
        keywords = data.get("Keywords", [])
        if keywords:
            all_keywords.extend(keywords)
        return {"Keywords": all_keywords} if all_keywords else None

    if isinstance(data, list):
        # List of dicts - extract Keywords from each
        for item in data:
            if isinstance(item, dict):
                keywords = item.get("Keywords", [])
                if keywords:
                    all_keywords.extend(keywords)
        return {"Keywords": all_keywords} if all_keywords else None

    if isinstance(data, str):
        # Try to find all JSON objects in the string
        # First, try to parse as single JSON
        try:
            parsed = json.loads(data)
            return clean_tags(parsed)  # Recursively handle the parsed result
        except:
            pass

        # Try to extract from JSON markdown
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, data, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return clean_tags(parsed)
            except:
                pass

        # Try to find multiple JSON objects in the string
        # Look for all {"Keywords": [...]} patterns
        keywords_pattern = r'"Keywords"\s*:\s*\[(.*?)\]'
        matches = re.findall(keywords_pattern, data, re.DOTALL)

        for match in matches:
            # Try to parse the array content
            try:
                # Reconstruct the JSON array and parse it
                array_str = '[' + match + ']'
                keywords = json.loads(array_str)
                if keywords:
                    all_keywords.extend(keywords)
            except:
                # If parsing fails, try with repair_json
                try:
                    array_str = '[' + match + ']'
                    keywords = json.loads(rj(array_str))
                    if keywords:
                        all_keywords.extend(keywords)
                except:
                    pass

        if all_keywords:
            return {"Keywords": all_keywords}

        # Last resort: try repair_json on the whole string
        try:
            parsed = json.loads(rj(data))
            return clean_tags(parsed)
        except:
            pass

    return None


class Config:
    def __init__(self):
        self.directory = None
        self.api_url = None
        self.api_password = None
        self.no_crawl = False
        self.no_backup = False
        self.dry_run = False
        self.update_keywords = False
        self.reprocess_failed = False
        self.reprocess_all = False
        self.reprocess_orphans = True
        self.text_completion = False
        self.gen_count = 250
        self.res_limit = 448
        self.detailed_caption = False
        self.short_caption = True
        self.skip_verify = False
        self.quick_fail = False
        self.no_caption = False
        self.update_caption = False
        self.use_sidecar = False
        self.normalize_keywords = True
        self.depluralize_keywords = False
        self.limit_word_count = True
        self.max_words_per_keyword = 2
        self.split_and_entries = True
        self.ban_prompt_words = True
        self.no_digits_start = True  
        self.min_word_length = True
        self.latin_only = True
        self.caption_instruction = "Describe the image. Be specific"
        self.system_instruction = "You are a helpful assistant."
        self.keyword_instruction = ""
        self.tag_instruction = 'Return a JSON object with key Keywords with the value as array of Keywords and tags that describe the image as follows: {"Keywords": []}' 

        # Sampler settings
        self.temperature = 0.2
        self.top_p = 1.0
        self.rep_pen = 1.01
        self.top_k = 100
        self.min_p = 0.05
        self.use_default_badwordsids = False
        self.use_json_grammar = False
        self.skip_folders = []
        self.rename_invalid = False
        self.preserve_date = False
        self.fix_extension = False
        #self.write_unsafe = False

        self.instruction = """Return a JSON object containing a Description for the image and a list of Keywords.

Write the Description using the active voice.

Generate 5 to 10 Keywords. Each Keyword is an item in a list and will be composed of a maximum of two words.

For both Description and Keywords, make sure to include:

 - Themes, concepts
 - Items, animals, objects
 - Structures, landmarks, setting
 - Foreground and background elements
 - Notable colors, textures, styles
 - Actions, activities

If humans are present, include:
 - Physical appearance
 - Gender
 - Clothing
 - Age range
 - Visibly apparent ancestry
 - Occupation/role
 - Relationships between individuals
 - Emotions, expressions, body language

Use ENGLISH only. Generate ONLY a JSON object with the keys Description and Keywords as follows {"Description": str, "Keywords": []}"""
        

        self.image_extensions = {
        "JPEG": [
            ".jpg",
            ".jpeg",
            ".jpe",
            ".jif",
            ".jfif",
            ".jfi",
            ".jp2",
            ".j2k",
            ".jpf",
            ".jpx",
            ".jpm",
            ".mj2",
        ],
        "PNG": [".png"],
        "GIF": [".gif"],
        "TIFF": [".tiff", ".tif"],
        "WEBP": [".webp"],
        "HEIF": [".heif", ".heic"],
        "RAW": [
            ".raw",  # Generic RAW
            ".arw",  # Sony
            ".cr2",  # Canon
            ".cr3",  # Canon (newer format)
            ".dng",  # Adobe Digital Negative
            ".nef",  # Nikon
            ".nrw",  # Nikon
            ".orf",  # Olympus
            ".pef",  # Pentax
            ".raf",  # Fujifilm
            ".rw2",  # Panasonic
            ".srw",  # Samsung
            ".x3f",  # Sigma
            ".erf",  # Epson
            ".kdc",  # Kodak
            ".rwl",  # Leica
        ]}
        
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Image Indexer")
        parser.add_argument("directory", help="Directory containing the files")
        parser.add_argument(
            "--api-url", default="http://localhost:5001", help="URL for the LLM API"
        )
        parser.add_argument(
            "--api-password", default="", help="Password for the LLM API"
        )
        parser.add_argument(
            "--no-crawl", action="store_true", help="Disable recursive indexing"
        )
        parser.add_argument(
            "--no-backup",
            action="store_true",
            help="Don't make a backup of files before writing",
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Don't write any files"
        )
        parser.add_argument(
            "--reprocess-all", action="store_true", help="Reprocess all files"
        )
        parser.add_argument(
            "--reprocess-failed", action="store_true", help="Reprocess failed files"
        )
        parser.add_argument(
            "--use-sidecar", action="store_true", help="Store generated data in an xmp sidecare instead of the image file"
        )
        parser.add_argument(
            "--reprocess-orphans", action="store_true", help="If a file has a UUID, determine its status"
        )
        parser.add_argument(
            "--update-keywords", action="store_true", help="Update existing keyword metadata"
        )
        parser.add_argument(
            "--gen-count", default=150, help="Number of tokens to generate"
        )
        parser.add_argument("--detailed-caption", action="store_true", help="Write a detailed caption along with keywords")
        parser.add_argument(
            "--skip-verify", action="store_true", help="Skip verifying file metadata validity before processing"
        )
        parser.add_argument("--update-caption", action="store_true", help="Add the generated caption to the existing description tag")
        parser.add_argument("--quick-fail", action="store_true", help="Mark failed after one try")
        parser.add_argument("--short-caption", action="store_true", help="Write a short caption along with keywords")
        parser.add_argument("--no-caption", action="store_true", help="Do not modify caption")
        parser.add_argument(
            "--normalize-keywords", action="store_true", help="Enable keyword normalization"
        )
        parser.add_argument("--res-limit", type=int, default=448, help="Limit the resolution of the image")
        parser.add_argument("--rename-invalid", type="store_true", help="Use rename invalid files so they don't get reprocessed")
        parser.add_argument("--preserve-date", type="store_true", help="Keep the original modified date, but will use a temp file when writing")
        args = parser.parse_args()

        config = cls()
        
        for key, value in vars(args).items():
            setattr(config, key, value)
        
        return config

class LLMProcessor:
    def __init__(self, config):
        self.api_url = config.api_url
        self.config = config
        self.instruction = config.instruction
        self.system_instruction = config.system_instruction
        self.caption_instruction = config.caption_instruction
        self.tag_instruction = config.tag_instruction
        self.requests = requests
        self.api_password = config.api_password
        self.max_tokens = config.gen_count
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.rep_pen = config.rep_pen
        self.top_k = config.top_k
        self.min_p = config.min_p
        self.use_default_badwordsids = config.use_default_badwordsids
        self.use_json_grammar = config.use_json_grammar

    def describe_content(self, task="", processed_image=None):
        if not processed_image:
            print("No image to describe.")

            return None

        # Determine instruction and whether to ban EOS token based on task
        if task == "caption":
            instruction = self.caption_instruction
            ban_eos = self.use_default_badwordsids

        elif task == "keywords":
            instruction = self.tag_instruction
            ban_eos = self.use_default_badwordsids  # Ban EOS token for keywords-only generation

        elif task == "caption_and_keywords":
            instruction = self.instruction
            ban_eos = self.use_default_badwordsids

        else:
            print(f"invalid task: {task}")

            return None

        try:
            messages = [
                {"role": "system", "content": self.system_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{processed_image}"
                            }
                        }
                    ]
                }
            ]

            payload = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "rep_pen": self.rep_pen,
                "use_default_badwordsids": ban_eos
            }

            # Add JSON schema if grammar is enabled and task requires structured output
            if self.use_json_grammar and task in ["caption_and_keywords", "keywords"]:
                if task == "caption_and_keywords":
                    # Schema for both description and keywords
                    payload["response_format"] = {
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "Description": {
                                    "type": "string"
                                },
                                "Keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": ["Description", "Keywords"]
                        }
                    }
                elif task == "keywords":
                    # Schema for keywords only
                    payload["response_format"] = {
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "Keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": ["Keywords"]
                        }
                    }

            endpoint = f"{self.api_url}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
            if self.api_password:
                headers["Authorization"] = f"Bearer {self.api_password}"

            response = self.requests.post(
                endpoint,
                json=payload,
                headers=headers
            )

            response.raise_for_status()
            response_json = response.json()

            if "choices" in response_json and len(response_json["choices"]) > 0:
                if "message" in response_json["choices"][0]:
                    content = response_json["choices"][0]["message"]["content"]
                    print(f"  Received response from API ({len(content)} chars)")
                    return content
                else:
                    content = response_json["choices"][0].get("text", "")
                    print(f"  Received response from API ({len(content)} chars)")
                    return content
            print(f"  Warning: API response missing expected data")
            return None
            
        except requests.exceptions.ConnectionError as e:
            print(f"API Connection Error: Cannot connect to {self.api_url}")
            print(f"  Make sure the LLM server is running and accessible")
            return None
        except requests.exceptions.Timeout as e:
            print(f"API Timeout Error: Request to {self.api_url} timed out")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"API HTTP Error: {e.response.status_code} - {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"  Response: {e.response.text[:200]}")
            return None
        except Exception as e:
            print(f"API Error: {type(e).__name__} - {str(e)}")
            return None

class BackgroundIndexer(threading.Thread):
    def __init__(self, root_dir, metadata_queue, file_extensions, no_crawl=False, chunk_size=100, skip_folders=None):
        threading.Thread.__init__(self)
        self.root_dir = root_dir
        self.metadata_queue = metadata_queue
        self.file_extensions = file_extensions
        self.no_crawl = no_crawl
        self.skip_folders = skip_folders if skip_folders else []
        self.total_files_found = 0
        self.indexing_complete = False
        self.chunk_size = chunk_size
        self.last_processed_dir = None

    def _should_skip_directory(self, directory):
        """Check if directory should be skipped based on skip_folders list"""
        if not self.skip_folders:
            return False

        # Normalize the directory path
        dir_normalized = os.path.normpath(directory)

        for skip_folder in self.skip_folders:
            skip_normalized = os.path.normpath(skip_folder)

            # Check if it's a full path match
            if dir_normalized == skip_normalized:
                return True

            # Check if it's a relative path from root_dir
            relative_skip = os.path.normpath(os.path.join(self.root_dir, skip_folder))
            if dir_normalized == relative_skip:
                return True

            # Check if the directory contains the skip folder in its path
            if skip_normalized in dir_normalized or os.path.basename(dir_normalized) == os.path.basename(skip_normalized):
                return True

        return False
            
    def run(self):
        if self.no_crawl:
            if not self._should_skip_directory(self.root_dir):
                print(f"Indexing directory (no crawl): {self.root_dir}")
                self._index_directory(self.root_dir)
        else:
            # Get ordered list of directories to process
            directories = []
            for root, _, _ in os.walk(self.root_dir):
                dir_path = os.path.normpath(root)
                if not self._should_skip_directory(dir_path):
                    directories.append(dir_path)

            directories.sort()
            print(f"Found {len(directories)} director(ies) to index")
            start_idx = 0

            for i in range(start_idx, len(directories)):
                self._index_directory(directories[i])

        print(f"Indexing complete. Total files found: {self.total_files_found}")
        self.indexing_complete = True

    def _index_directory(self, directory):
        """Process directory in chunks"""
        directory = os.path.normpath(directory)
        file_batch = []
        
        try:
            for filename in os.listdir(directory):
                file_path = os.path.normpath(os.path.join(directory, filename))
                
                # Skip if not a valid file type
                if not any(file_path.lower().endswith(ext) for ext in self.file_extensions):
                    continue
                        
                try:
                    # Check for 0 byte files
                    size = os.path.getsize(file_path)
                    if size > 0:
                        file_batch.append(file_path)
                        
                        # When we reach chunk size, send batch to queue
                        if len(file_batch) >= self.chunk_size:
                            self.total_files_found += len(file_batch)
                            self.metadata_queue.put((directory, file_batch))
                            file_batch = []
                            
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                    
            # Don't forget the last batch
            if file_batch:
                self.total_files_found += len(file_batch)
                self.metadata_queue.put((directory, file_batch))
                
        except (PermissionError, OSError):
            print(f"Permission denied or error accessing directory: {directory}")


class FileProcessor:
    def __init__(self, config, check_paused_or_stopped=None, callback=None):
        self.config = config
        self.llm_processor = LLMProcessor(config)
        
        if check_paused_or_stopped is None:
            self.check_paused_or_stopped = lambda: False
        else:
            self.check_paused_or_stopped = check_paused_or_stopped
            
        if callback is None:
            self.callback = print
        else:
            self.callback = callback
        self.failed_validations = []
        self.files_in_queue = 0
        self.total_processing_time = 0
        self.files_processed = 0
        self.files_completed = 0
        
        self.image_processor = ImageProcessor(max_dimension=self.config.res_limit, patch_sizes=[14])

        print("Initializing ExifTool...")
        self.et = exiftool.ExifToolHelper(encoding='utf-8')
        print("ExifTool initialized successfully")
        
        # Words in the prompt tend to get repeated back by certain models
        self.banned_words = ["no", "unspecified", "unknown", "unidentified", "identify", "topiary", "themes concepts", "items animals", "animals objects", "structures landmarks", "Foreground and background", "notable colors", "textures styles", "actions activities", "physical appearance", "Gender", "Age range", "visibly apparent", "apparent ancestry", "Occupation/role", "Relationships between individuals", "Emotions expressions", "body language"]
                
        self.keyword_fields = [
            "Keywords",
            "IPTC:Keywords",
            "Composite:keywords",
            "Subject",
            "DC:Subject",
            "XMP:Subject",
            "XMP-dc:Subject"
        ]
        self.caption_fields = [
            "Description",
            "XMP:Description",
            "ImageDescription",
            "DC:Description",
            "EXIF:ImageDescription",
            "Composite:Description",
            "Caption",
            "IPTC:Caption",
            "Composite:Caption"
            "IPTC:Caption-Abstract",
            "XMP-dc:Description",
            "PNG:Description"
        ]

        self.identifier_fields = [
            "Identifier",
            "XMP:Identifier",            
        ]
        self.status_fields = [
            "Status",
            "XMP:Status"
        ]
        self.filetype_fields = [
            "File:FileType",
            "File:FileTypeExtension"
        ]
        
        self.image_extensions = config.image_extensions
        self.metadata_queue = queue.Queue()

        chunk_size = getattr(config, 'chunk_size', 100)
        skip_folders = getattr(config, 'skip_folders', [])

        self.indexer = BackgroundIndexer(
            config.directory,
            self.metadata_queue,
            [ext for exts in self.image_extensions.values() for ext in exts],
            config.no_crawl,
            chunk_size=chunk_size,
            skip_folders=skip_folders
        )
        
        self.indexer.start()

    def rename_to_invalid(self, file_path):
        """ Rename a file to filename_ext.invalid
            Returns True if successful, False otherwise
        """
        try:
            # Clean up any exiftool temporary and backup files first
            dir_name = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)

            # Look for various exiftool temporary file patterns
            # Standard pattern: filename_exiftool_tmp
            exiftool_tmp = file_path + "_exiftool_tmp"
            if os.path.exists(exiftool_tmp):
                try:
                    os.remove(exiftool_tmp)
                    self.callback(f"Cleaned up temporary file: {os.path.basename(exiftool_tmp)}")
                except Exception as e:
                    self.callback(f"Could not remove temp file {os.path.basename(exiftool_tmp)}: {str(e)}")

            # Check for backup files created by exiftool when -overwrite_original is not used
            # These have _original suffix
            backup_file = file_path + "_original"
            if os.path.exists(backup_file):
                # If the main file doesn't exist but the backup does, this IS the file to rename
                if not os.path.exists(file_path):
                    file_path = backup_file
                    base_name = os.path.basename(backup_file)
                else:
                    # Both exist - remove the backup
                    try:
                        os.remove(backup_file)
                        self.callback(f"Cleaned up backup file: {os.path.basename(backup_file)}")
                    except Exception as e:
                        self.callback(f"Could not remove backup file <{os.path.basename(backup_file)}>: {str(e)}")

            # Check if the file to rename exists
            if not os.path.exists(file_path):
                self.callback(f"File no longer exists, cannot rename: {base_name}")
                return False

            # Replace dots with underscores except the last one, then add .invalid
            # Format: filename_ext.invalid or filename_ext(N).invalid for duplicates
            name_parts = base_name.rsplit('.', 1)
            if len(name_parts) == 2:
                base_invalid_name = f"{name_parts[0]}_{name_parts[1]}"
            else:
                base_invalid_name = base_name

            new_path = os.path.join(dir_name, f"{base_invalid_name}.invalid")

            # If a file with this name already exists, add a counter in parentheses
            counter = 1
            original_new_path = new_path
            while os.path.exists(new_path):
                new_name_with_counter = f"{base_invalid_name}({counter}).invalid"
                new_path = os.path.join(dir_name, new_name_with_counter)
                counter += 1
                # Safety limit to avoid infinite loop
                if counter > 1000:
                    self.callback(f"Too many duplicate .invalid files, cannot rename: {base_name}")
                    return False

            # Rename the file
            os.rename(file_path, new_path)
            if new_path != original_new_path:
                self.callback(f"Renamed invalid file: {base_name} -> {os.path.basename(new_path)} (duplicate name)")
                print(f"Invalid or corrupt file <{base_name}> renamed to <{os.path.basename(new_path)}> (duplicate name)")
            else:
                self.callback(f"Renamed invalid file: {base_name} -> {os.path.basename(new_path)}")
                print(f"Invalid or corrupt file <{base_name}> renamed to <{os.path.basename(new_path)}>")
            return True
        except Exception as e:
            self.callback(f"Failed to rename invalid file <{file_path}>: {str(e)}")
            print(f"Failed: {str(e)}")
            return False

    def fix_file_extension(self, file_path, expected_ext):
        """ Fix file extension if it doesn't match the expected extension from metadata.
            Returns the new file path if renamed, or the original path if no change needed.
        """
        if not expected_ext:
            return file_path

        # Normalize the expected extension (lowercase, with leading dot)
        expected_ext = expected_ext.lower()
        if not expected_ext.startswith('.'):
            expected_ext = '.' + expected_ext

        # Get current extension
        current_ext = os.path.splitext(file_path)[1].lower()

        # If they match, no change needed
        if current_ext == expected_ext:
            return file_path

        try:
            # Build new path with correct extension
            base_path = os.path.splitext(file_path)[0]
            new_path = base_path + expected_ext

            # Check if target path already exists
            counter = 1
            original_new_path = new_path
            while os.path.exists(new_path):
                new_path = f"{base_path}({counter}){expected_ext}"
                counter += 1
                if counter > 1000:
                    self.callback(f"Too many files with same name, cannot rename: {os.path.basename(file_path)}")
                    return file_path

            # Rename the file
            os.rename(file_path, new_path)

            if new_path != original_new_path:
                print(f"Fixed extension: {os.path.basename(file_path)} -> {os.path.basename(new_path)} (duplicate name)")
                self.callback(f"Fixed extension: {os.path.basename(file_path)} -> {os.path.basename(new_path)}")
            else:
                print(f"Fixed extension: {os.path.basename(file_path)} -> {os.path.basename(new_path)}")
                self.callback(f"Fixed extension: {os.path.basename(file_path)} -> {os.path.basename(new_path)}")

            return new_path

        except Exception as e:
            self.callback(f"Failed to fix extension for {file_path}: {str(e)}")
            print(f"Extension fix failed: {str(e)}")
            return file_path

    def process_directory(self, directory):
        try:
            while not (self.indexer.indexing_complete and self.metadata_queue.empty()):
                if self.check_pause_stop():
                    return
                
                try:
                    directory, files = self.metadata_queue.get(timeout=1)
                    self.callback(f"Processing directory: {directory}")
                    self.callback(f"---")

                    batch_size = 50
                    for i in range(0, len(files), batch_size):
                        batch = files[i:i+batch_size]
                        if len(batch) > 0:
                            print(f"Reading metadata for {len(batch)} file(s)...")
                        metadata_list = self._get_metadata_batch(batch)

                        for metadata in metadata_list:
                            if metadata:
                                # Process metadata
                                keywords = []
                                status = None
                                identifier = None
                                caption = None
                                validation_data = None

                                # Make a copy with only the fields we want to write
                                new_metadata = {}

                                # Check if we actually have a sidecar in the path
                                if self.config.use_sidecar and metadata["SourceFile"].lower().endswith(".xmp"):
                                    metadata["SourceFile"] = os.path.splitext(metadata["SourceFile"])[0]

                                new_metadata["SourceFile"] = metadata.get("SourceFile")

                                # Extract validation data if present
                                if not self.config.skip_verify and "ExifTool:Validate" in metadata:
                                    validation_data = metadata.get("ExifTool:Validate")

                                filetype = None
                                filetype_ext = None
                                for key, value in metadata.items():
                                    if key in self.keyword_fields:
                                        keywords.extend(value)
                                    if key in self.caption_fields:
                                        caption = value
                                    if key in self.identifier_fields:
                                        identifier = value
                                    if key in self.status_fields:
                                        status = value
                                    if key == "File:FileType":
                                        filetype = value
                                    if key == "File:FileTypeExtension":
                                        filetype_ext = value

                                # Standardize the fields
                                if keywords:
                                    new_metadata["MWG:Keywords"] = keywords
                                if caption:
                                    new_metadata["MWG:Description"] = caption
                                if status:
                                    new_metadata["XMP:Status"] = status
                                if identifier:
                                    new_metadata["XMP:Identifier"] = identifier
                                if validation_data:
                                    new_metadata["ExifTool:Validate"] = validation_data
                                if filetype:
                                    new_metadata["File:FileType"] = filetype
                                if filetype_ext:
                                    new_metadata["File:FileTypeExtension"] = filetype_ext

                                self.files_processed += 1

                                self.process_file(new_metadata)

                            if self.check_pause_stop():
                                return

                    self.update_progress()

                except queue.Empty:
                    continue
        finally:
            try:
                self.et.terminate()
                self.callback("ExifTool process terminated cleanly")
                
            except Exception as e:
                self.callback(f"Warning: ExifTool termination error: {str(e)}")

            
        
    def get_file_type(self, file_ext):
        """ If the filetype is supported, return the key
            so .nef would return RAW. Otherwise return
            None.
        """
        if not file_ext.startswith("."):
            file_ext = "." + file_ext
        
        file_ext = file_ext.lower()
        
        for file_type, extensions in self.image_extensions.items():
            if file_ext in [ext.lower() for ext in extensions]:
                
                return file_type
        
        return None

    def check_uuid(self, metadata, file_path):
        """ Very important or we end up processing 
            files more than once
        """ 
        try:
            status = metadata.get("XMP:Status")
            identifier = metadata.get("XMP:Identifier")
            keywords = metadata.get("MWG:Keywords")
            caption = metadata.get("MWG:Description")
            
            # Orphan check
            if identifier and self.config.reprocess_orphans and keywords and not status:
                    metadata["XMP:Status"] = "success"                    
                    status = "success"
                    try:
                        written = self.write_metadata(file_path, metadata)
                        
                        if written and not self.config.reprocess_all:
                            
                            print(f"Status added for orphan: {file_path}")  
                            self.callback(f"Status added for orphan: {file_path}")
                            
                        else:
                            print(f"Metadata write error for orphan: {file_path}")
                            self.callback(f"Metadata write error for orphan: {file_path}")
                            return None
                    except:
                        print("Error writing orphan status")
                        return None
        
            # Does file have a UUID in metadata
            if identifier:
                if not self.config.reprocess_all and status == "success":
                    
                    return None
                    
                # If it is retry, do it again
                if self.config.reprocess_all or status == "retry":
                    metadata["XMP:Status"] = None
                    
                    return metadata
                
                # If it is fail, don't do it unless we specifically want to
                if status == "failed":
                    if self.config.reprocess_failed or self.config.reprocess_all:
                        metadata["XMP:Status"] = None
                        
                        return metadata                    
                    
                    else:
                        return None
                
                # If there are no keywords, processs it                
                if not keywords:
                    metadata["XMP:Status"] = None
                    
                    return metadata
                
                else:
                    return None
                
            # No UUID, treat as new file
            else:
                metadata["XMP:Identifier"] = str(uuid.uuid4())
                
                return metadata  # New file

        except Exception as e:
            print(f"Error checking UUID: {str(e)}")
            
            return None
                        
    def check_pause_stop(self):
        if self.check_paused_or_stopped():
            
            while self.check_paused_or_stopped():
                time.sleep(0.1)
            
            if self.check_paused_or_stopped():
                return True
        
        return False

    def _get_metadata_batch(self, files):
        """ Get metadata for a batch of files
            using persistent ExifTool instance.
        """
        exiftool_fields = self.keyword_fields + self.caption_fields + self.identifier_fields + self.status_fields + self.filetype_fields
        
        try:
            if self.config.skip_verify:
                params = []
            else:
                params = ["-validate"]   
            
            # Use sidecars if they exist for metadata instead of images because
            # that is where we will have put the UUID and Status info
            if self.config.use_sidecar:
                xmp_files = []
                for file in files:
                    
                    # Check for files named file.ext.xmp for sidecar
                    if os.path.exists(file + ".xmp"):
                       xmp_files.append(file  + ".xmp")

                    else:
                        xmp_files.append(file)
                files = xmp_files
            return self.et.get_tags(files, tags=exiftool_fields, params=params)
            
        except exiftool.exceptions.ExifToolExecuteError as e:
            print(f"ExifTool Execute Error: {str(e)}")
            self.callback(f"ExifTool execute error - check if files are accessible")
            return []
        except exiftool.exceptions.ExifToolVersionError as e:
            print(f"ExifTool Version Error: {str(e)}")
            print("  Please update ExifTool to a compatible version")
            return []
        except Exception as e:
            print(f"ExifTool Error: {type(e).__name__} - {str(e)}")
            return []

    def update_progress(self):
        files_processed = self.files_processed
        files_remaining = self.indexer.total_files_found - files_processed
        
        if files_remaining < 0:
            files_remaining = 0
        
        self.callback(f"Directory processed. Files remaining in queue: {files_remaining}")
        self.callback(f"---")
        
    
    def process_file(self, metadata):
        """ Process a file and update its metadata in one operation.
            This minimizes the number of writes to the file.
        """
        try:

            success = True
            file_path = metadata["SourceFile"]

            # If the file doesn't exist anymore, skip it
            if not os.path.exists(file_path):
                self.callback(f"File no longer exists: {file_path}")
                self.callback(f"---")
                return

            # Check if file is already marked as invalid - skip it entirely
            # Unless reprocess_all is enabled
            current_status = metadata.get("XMP:Status")
            if current_status == "invalid" and not self.config.reprocess_all:
                self.callback(f"Skipping file marked as invalid: {file_path}")
                self.callback(f"---")
                return

            # Only run validation check for files without a status or if reprocess_all
            # This prevents re-validating files that are already success/failed/retry/valid
            should_validate = (not current_status or self.config.reprocess_all) and not self.config.skip_verify

            if should_validate:
                validation_parts = metadata.get("ExifTool:Validate", "0 0 0").split()
                if len(validation_parts) >= 3:
                    errors, warnings, minor = map(int, validation_parts[:3])
                else:
                    errors, warnings, minor = 0, 0, 0

                # If there are validation errors, mark as invalid and skip
                if errors > 0:
                    print(f"Validation Failed: {os.path.basename(file_path)}")
                    print(f"  Errors: {errors}, Warnings: {warnings}, Minor: {minor}")
                    self.callback(f"\nValidation failed: {file_path}")
                    self.callback(f"  Errors: {errors}, Warnings: {warnings}, Minor: {minor}")
                    self.failed_validations.append(file_path)
                    if self.config.rename_invalid:
                        self.rename_to_invalid(file_path)
                    self.callback(f"---")
                    return

                # If there are warnings, test if we can write to the file
                # This prevents wasting LLM processing on unwritable files
                if (warnings > 0) and (minor >= warnings):
                    print(f"File has validation warnings: {os.path.basename(file_path)}")
                    print(f"  Warnings: {warnings}, Minor: {minor} - Testing writeability...")
                    test_metadata = {"SourceFile": file_path, "XMP:Status": "valid"}
                    if not self.write_metadata(file_path, test_metadata):
                        print(f"  Metadata cannot be written to file")
                        self.callback(f"\nMetadata is not writable: {file_path}")
                        self.failed_validations.append(file_path)
                        self.callback(f"---")
                        return
                    print(f"  File is writable - proceeding")
                    # File is writable, update metadata to reflect valid status
                    metadata["XMP:Status"] = "valid"

            metadata = self.check_uuid(metadata, file_path)
            if not metadata:
                return
                
            image_type = self.get_file_type(os.path.splitext(file_path)[1].lower())
            if image_type is None:
                self.callback(f"Not a supported image type: {file_path}")
                self.callback(f"---")
                return

            filetype = metadata.get("File:FileType", image_type)
            print(f"Processing: {os.path.basename(file_path)} [{filetype}]")

            start_time = time.time()

            try:
                processed_image, image_path = self.image_processor.process_image(file_path)
            except Exception as e:
                print(f"Image Processing Error: {os.path.basename(file_path)}")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Details: {str(e)}")
                self.callback(f"Image processing error for {file_path}: {str(e)}")
                if self.config.rename_invalid:
                    self.rename_to_invalid(file_path)
                self.callback(f"---")
                return

            if not processed_image:
                print(f"Image Processing Failed: {os.path.basename(file_path)}")
                print(f"  Could not generate base64 image data")
                self.callback(f"Failed to process image: {file_path}")
                if self.config.rename_invalid:
                    self.rename_to_invalid(file_path)
                self.callback(f"---")
                return

            updated_metadata = self.generate_metadata(metadata, processed_image)
           
            status = updated_metadata.get("XMP:Status")
            
            # Retry one time if failed
            if not self.config.quick_fail and status == "retry":
                print(f"AI Generation Issue - Retrying: {os.path.basename(file_path)}")
                print(f"  Reason: No valid keywords generated on first attempt")
                self.callback(f"Asking AI to try again for {file_path}...")
                self.callback(f"---")
                updated_metadata = self.generate_metadata(metadata, processed_image)
                status = updated_metadata.get("XMP:Status")

            # If retry didn't work, mark failed
            if not status == "success":
                print(f"AI Generation Failed: {os.path.basename(file_path)}")
                print(f"  The AI could not generate valid keywords after retry")
                self.callback(f"Retry failed due to AI for {file_path}")
                self.callback(f"---")
                metadata["XMP:Status"] = "failed"
                
                if not self.config.dry_run:
                    success = False
                    self.write_metadata(file_path, metadata)
                
                
            # Fix file extension if enabled (before writing metadata)
            if self.config.fix_extension and success:
                expected_ext = metadata.get("File:FileTypeExtension")
                if expected_ext:
                    new_file_path = self.fix_file_extension(file_path, expected_ext)
                    if new_file_path != file_path:
                        file_path = new_file_path
                        updated_metadata["SourceFile"] = file_path

            # Send image data to callback for GUI display
            if self.callback and hasattr(self.callback, '__call__') and success:

                # Create a dictionary with image data for GUI
                image_data = {
                    'type': 'image_data',
                    'base64_image': processed_image,
                    'caption': updated_metadata.get('MWG:Description', ''),
                    'keywords': updated_metadata.get('MWG:Keywords', []),
                    'file_path': file_path
                }

                self.callback(image_data)

            if not self.config.dry_run and success:
                write_success = self.write_metadata(file_path, updated_metadata)
                if write_success:
                    print(f"  Metadata written successfully")
                    success = True
                else:
                    success = False
                    #print(f"Could not write new metadata to file: {file_path}") 
                    #self.callback(f"Failed writing metadata for {file_path}")
                    #self.callback(f"---")
                    
            end_time = time.time()
            processing_time = end_time - start_time
            self.total_processing_time += processing_time
            self.files_completed += 1
            
            # Calculate and display progress info
            in_queue = self.indexer.total_files_found - self.files_processed
            average_time = self.total_processing_time / self.files_completed
            time_left = average_time * in_queue
            time_left_unit = "s"
            
            if time_left > 180:
                time_left = time_left / 60
                time_left_unit = "mins"
            
            if time_left < 0:
                time_left = 0
            
            if in_queue < 0:
                in_queue = 0
            if success:
                 
                self.callback(f"<b>Image:</b> {os.path.basename(file_path)}")
                self.callback(f"<b>Status:</b> {status}")

                self.callback(
                    f"<b>Processing time:</b> {processing_time:.2f}s, <b>Average processing time:</b> {average_time:.2f}s"
                )
                self.callback(
                    f"<b>Processed:</b> {self.files_processed}, <b>In queue:</b> {in_queue}, <b>Time remaining (est):</b> {time_left:.2f}{time_left_unit}"
                )
                self.callback("---")   
                
            if self.check_pause_stop():
                return
            
        except Exception as e:
            print(f"Processing Error: {os.path.basename(file_path)}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Details: {str(e)}")
            self.callback(f"<b>Error processing:</b> {file_path}: {str(e)}")
            self.callback(f"---")
            return
    
    def generate_metadata(self, metadata, processed_image):
        """ Generate metadata without writing to file.
            Returns (metadata_dict)
            
            short_caption will get a short caption in a single generation
            
            detailed_caption will get get a detailed caption using two
            generations
            
            update_caption appends new caption to existing caption to the existing description.
            
        """
        new_metadata = {}
        existing_caption = metadata.get("MWG:Description")
        caption = None
        keywords = None
        detailed_caption = ""
        old_keywords = metadata.get("MWG:Keywords", [])
        file_path = metadata["SourceFile"]
        
        try:

            # Determine whether to generate caption, keywords, or both
            if not self.config.no_caption and self.config.detailed_caption:
                print(f"  Generating keywords and detailed caption...")
                data = clean_tags(self.llm_processor.describe_content(task="keywords", processed_image=processed_image))
                detailed_caption = clean_string(self.llm_processor.describe_content(task="caption", processed_image=processed_image))               
                
                if existing_caption and self.config.update_caption:
                    caption = existing_caption + "<generated>" + detailed_caption + "</generated>"
                
                else:
                    caption = detailed_caption
                
                if isinstance(data, dict):
                    keywords = data.get("Keywords")
                   
            else:
                if self.config.no_caption:
                    print(f"  Generating keywords only...")
                    data = clean_tags(self.llm_processor.describe_content(task="keywords", processed_image=processed_image))
                else:
                    print(f"  Generating caption and keywords...")
                    data = clean_json(self.llm_processor.describe_content(task="caption_and_keywords", processed_image=processed_image))
                         
                if isinstance(data, dict):
                    keywords = data.get("Keywords")
                
                    if not existing_caption and not self.config.no_caption:
                        caption = data.get("Description")
                    
                    elif existing_caption and self.config.update_caption:
                        caption = existing_caption + "<generated>" + data.get("Description") + "</generated>"
                    
                    elif data.get("Description") and not self.config.no_caption:
                        caption = data.get("Description")
                    
                    else:
                        if existing_caption:
                            caption = existing_caption
                        else:
                            caption = ""
                        
            if not keywords:
                print(f"No Keywords Generated: {os.path.basename(file_path)}")
                print(f"  AI response did not contain valid keywords")
                status = "retry"

            else:
                status = "success"
                keywords = self.process_keywords(metadata, keywords)
                if keywords:
                    print(f"Generated {len(keywords)} keyword(s) for: {os.path.basename(file_path)}")

            new_metadata["MWG:Description"] = caption
            new_metadata["MWG:Keywords"] = keywords
            new_metadata["XMP:Status"] = status
            new_metadata["XMP:Identifier"] = metadata.get("XMP:Identifier", str(uuid.uuid4()))
            new_metadata["SourceFile"] = file_path
            
            return new_metadata
            
        except Exception as e:
            print(f"Metadata Generation Error: {os.path.basename(file_path)}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Details: {str(e)}")
            self.callback(f"Parse error for {file_path}: {str(e)}")
            self.callback(f"---")
            metadata["XMP:Status"] = "retry"

            return metadata
            
    def write_metadata(self, file_path, metadata):
        """Write metadata using persistent ExifTool instance"""
        if self.config.dry_run:
            print("Dry run. Not writing.")

            return True

        # Keep track of original file path for error handling
        original_file_path = file_path

        try:
            # -m: ignore minor errors
            params = ["-m"]

            # -P: preserve file modification date (can cause temp files)
            if self.config.preserve_date:
                params.append("-P")

            # Overwrite in place to avoid temp files when no_backup is set, or when using sidecar
            if self.config.no_backup or self.config.use_sidecar:
                params.append("-overwrite_original")

            if self.config.use_sidecar:
                file_path = file_path + ".xmp"
            #if self.config.write_unsafe:
                #params.append("-unsafe")
            # Use existing ExifTool instance
            self.et.set_tags(file_path, tags=metadata, params=params)

            return True

        except Exception as e:
            error_type = type(e).__name__
            print(f"Metadata Write Error: {os.path.basename(original_file_path)}")
            print(f"  Error type: {error_type}")
            print(f"  Details: {str(e)}")
            self.callback(f"\nError writing metadata: {str(e)}")
            if self.config.rename_invalid:
                # Rename the original image file, not the sidecar
                print(f"  Renaming file to .invalid")
                self.rename_to_invalid(original_file_path)
                # Also clean up the sidecar if it exists
                if self.config.use_sidecar:
                    sidecar_path = original_file_path + ".xmp"
                    if os.path.exists(sidecar_path):
                        try:
                            os.remove(sidecar_path)
                            self.callback(f"Removed incomplete sidecar file")
                        except:
                            pass
            #print(f"\nError: {str(e)}")
            #self.callback(f"---")
            return False 
    
    def process_keywords(self, metadata, new_keywords):
        """ Normalize extracted keywords and deduplicate them.
            If update is configured, combine the old and new keywords.
        """
        all_keywords = set()
              
        if self.config.update_keywords:
            existing_keywords = metadata.get("MWG:Keywords", [])

            if isinstance(existing_keywords, str):
                existing_keywords = [k.strip() for k in existing_keywords.split(",")]
                
            for keyword in existing_keywords:
                normalized = normalize_keyword(keyword, self.banned_words, self.config)
            
                if normalized:
                    all_keywords.add(normalized)
                           
        for keyword in new_keywords:
            normalized = normalize_keyword(keyword, self.banned_words, self.config)
            
            if normalized:
                all_keywords.add(normalized)

        if all_keywords:        
            return list(all_keywords)
        else:
            return None
        
def main(config=None, callback=None, check_paused_or_stopped=None):
    if config is None:
        config = Config.from_args()
    
    if not hasattr(config, 'chunk_size'):
        config.chunk_size = 100
             
    file_processor = FileProcessor(
        config, check_paused_or_stopped, callback
    )      
    
    try:
        file_processor.process_directory(config.directory)
    
    except KeyboardInterrupt:
        print("Processing interrupted. State saved for resuming later.")
        
        if callback:
            callback("Processing interrupted. State saved for resuming later.")
    
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
    
        if callback:
            callback(f"Error: {str(e)}")
            
    finally:
        print("Waiting for indexer to complete...")
        file_processor.indexer.join()
        print("Indexing completed.")
        
   
if __name__ == "__main__":
    main()
