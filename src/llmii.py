import os, json, time, re, argparse, exiftool, threading, queue, calendar, io, uuid, requests
from json_repair import repair_json as rj
from datetime import timedelta
from .image_processor import ImageProcessor
from .llmii_utils import first_json, de_pluralize, AND_EXCEPTIONS

class SkipDirectoryException(Exception):
    """Exception raised to skip the current directory"""
    pass
    
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
    if config.no_digits_start and re.match(r'^\d{3,}', words[0]):
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
    
    # More than one think tag
    #if len(data.split("<think>")) > 2:
    #    return none
        
    # One think tag, remove everything before it.
    if "<think>" in data:
        data = data[data.index("<think>")+7:]
    
    # Remove everything after an end think tag
    data = data.split("</think>")[0]
    
    if isinstance(data, str):
        data = re.sub(r"\n", "", data)
        data = re.sub(r'["""]', '"', data)
        data = re.sub(r"\\{2}", "", data)
        last_period = data.rfind('.')
        
        if last_period != -1:
            data = data[:last_period+1]
    
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
                    return response_json["choices"][0]["message"]["content"]
                else:
                    return response_json["choices"][0].get("text", "")
            return None
            
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return None

class BackgroundIndexer(threading.Thread):
    def __init__(self, root_dir, metadata_queue, file_extensions, no_crawl=False, chunk_size=100):
        threading.Thread.__init__(self)
        self.root_dir = root_dir
        self.metadata_queue = metadata_queue
        self.file_extensions = file_extensions
        self.no_crawl = no_crawl
        self.total_files_found = 0
        self.indexing_complete = False
        self.chunk_size = chunk_size
        self.checkpoint_path = os.path.join(root_dir, ".llmii_checkpoint")
        self.last_processed_dir = self._load_checkpoint()
        
    def _load_checkpoint(self):
        """Load directory checkpoint if it exists"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    return f.read().strip()
            except:
                return None
        return None
        
    def _save_checkpoint(self, directory):
        """Save current directory as checkpoint"""
        try:
            with open(self.checkpoint_path, 'w') as f:
                f.write(directory)
        except:
            pass
            
    def run(self):
        if self.no_crawl:
            self._index_directory(self.root_dir)
        else:
            # Get ordered list of directories to process
            directories = []
            for root, _, _ in os.walk(self.root_dir):
                directories.append(os.path.normpath(root))
            
            directories.sort()
            
            # Skip to last processed directory if resuming
            start_idx = 0
            if self.last_processed_dir:
                try:
                    start_idx = directories.index(self.last_processed_dir) + 1
                    # If already completed the last directory, just start from beginning
                    if start_idx >= len(directories):
                        start_idx = 0
                except ValueError:
                    start_idx = 0
                    
            for i in range(start_idx, len(directories)):
                self._index_directory(directories[i])
                self._save_checkpoint(directories[i])
                
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
        
        self.files_in_queue = 0
        self.total_processing_time = 0
        self.files_processed = 0
        self.files_completed = 0
        
        self.image_processor = ImageProcessor(max_dimension=self.config.res_limit, patch_sizes=[14])
        
        self.et = exiftool.ExifToolHelper(encoding='utf-8')
        
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
        
        self.image_extensions = config.image_extensions
        self.metadata_queue = queue.Queue()

        chunk_size = getattr(config, 'chunk_size', 100)
        
        self.indexer = BackgroundIndexer(
            config.directory, 
            self.metadata_queue, 
            [ext for exts in self.image_extensions.values() for ext in exts], 
            config.no_crawl,
            chunk_size=chunk_size
        )
        
        self.file_checkpoint_path = os.path.join(config.directory, ".llmii_file_checkpoint")
        self.last_processed_file = self._load_file_checkpoint()
        
        self.indexer.start()
        
    def _load_file_checkpoint(self):
        """Load file processing checkpoint if it exists"""
        if os.path.exists(self.file_checkpoint_path):
            try:
                with open(self.file_checkpoint_path, 'r') as f:
                    return f.read().strip()
            except:
                return None
        return None
        
    def _save_file_checkpoint(self, file_path):
        """Save current file as checkpoint"""
        try:
            with open(self.file_checkpoint_path, 'w') as f:
                f.write(file_path)
        except:
            pass
    
    def process_directory(self, directory):
        try:
            while not (self.indexer.indexing_complete and self.metadata_queue.empty()):
                if self.check_pause_stop():
                    return
                
                try:
                    directory, files = self.metadata_queue.get(timeout=1)
                    self.callback(f"Processing directory: {directory}")
                    self.callback(f"---")

                    try:
                        # If we have a last processed file checkpoint, find where to resume
                        if self.last_processed_file:
                            try:
                                resume_idx = files.index(self.last_processed_file) + 1
                                # Only skip if we haven't processed the entire directory yet
                                if resume_idx < len(files):
                                    self.callback(f"Resuming from file: {self.last_processed_file}")
                                    files = files[resume_idx:]
                                self.last_processed_file = None  # Reset after finding
                            except ValueError:
                                # File not in this batch, process normally
                                pass

                        batch_size = 50
                        for i in range(0, len(files), batch_size):
                            batch = files[i:i+batch_size]
                            metadata_list = self._get_metadata_batch(batch)

                            for metadata in metadata_list:
                                if metadata:
                                    if not self.config.skip_verify:
                                        # Check ExifTool validation
                                        if "ExifTool:Validate" in metadata:
                                            errors, warnings, minor = map(int, metadata.get("ExifTool:Validate", "0 0 0").split())
                                            source_file = metadata.get("SourceFile")

                                            if errors > 0:
                                                print(f"{source_file}: failed to validate. Skipping!")
                                                self.callback(f"\n{source_file}: failed to validate. Skipping!")
                                                self.callback(f"---")
                                                self.files_processed +=1
                                                continue

                                    # Process metadata
                                    keywords = []
                                    status = None
                                    identifier = None
                                    caption = None

                                    # Make a copy with only the fields we want to write
                                    new_metadata = {}

                                    # Check if we actually have a sidecar in the path
                                    if self.config.use_sidecar and metadata["SourceFile"].lower().endswith(".xmp"):
                                        metadata["SourceFile"] = os.path.splitext(metadata["SourceFile"])[0]

                                    new_metadata["SourceFile"] = metadata.get("SourceFile")

                                    for key, value in metadata.items():
                                        if key in self.keyword_fields:
                                            keywords.extend(value)
                                        if key in self.caption_fields:
                                            caption = value
                                        if key in self.identifier_fields:
                                            identifier = value
                                        if key in self.status_fields:
                                            status = value

                                    # Standardize the fields
                                    if keywords:
                                        new_metadata["MWG:Keywords"] = keywords
                                    if caption:
                                        new_metadata["MWG:Description"] = caption
                                    if status:
                                        new_metadata["XMP:Status"] = status
                                    if identifier:
                                        new_metadata["XMP:Identifier"] = identifier

                                    self.files_processed += 1

                                    # Save checkpoint before processing file
                                    self._save_file_checkpoint(new_metadata["SourceFile"])

                                    self.process_file(new_metadata)

                                    # Clear checkpoint after successful processing
                                    if os.path.exists(self.file_checkpoint_path):
                                        os.remove(self.file_checkpoint_path)

                                if self.check_pause_stop():
                                    return

                        self.update_progress()

                    except SkipDirectoryException as e:
                        self.callback(f"Skipped directory: {directory}")
                        self.callback(f"---")
                        continue

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
        exiftool_fields = self.keyword_fields + self.caption_fields + self.identifier_fields + self.status_fields 
        
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
            
        except Exception as e:
            print("Exiftool error")
            
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
            
            
            file_path = metadata["SourceFile"]
                
            # If the file doesn't exist anymore, skip it
            if not os.path.exists(file_path):
                self.callback(f"File no longer exists: {file_path}")
                self.callback(f"---")
                return
            
            metadata = self.check_uuid(metadata, file_path)
            if not metadata:
                return
                
            image_type = self.get_file_type(os.path.splitext(file_path)[1].lower())
            if image_type is None:
                self.callback(f"Not a supported image type: {file_path}")
                self.callback(f"---")
                return
                
            start_time = time.time()
            
            processed_image, image_path = self.image_processor.process_image(file_path)
            updated_metadata = self.generate_metadata(metadata, processed_image)
           
            status = updated_metadata.get("XMP:Status")
            
            # Retry one time if failed
            if not self.config.quick_fail and status == "retry":
                print(f"Retrying {file_path} once")
                self.callback(f"Retrying {file_path}...")
                self.callback(f"---")
                updated_metadata = self.generate_metadata(metadata, processed_image)      
                status = updated_metadata.get("XMP:Status")
            
            # If retry didn't work, mark failed
            if not status == "success":
                print(f"failed: {file_path}")
                self.callback(f"Retry failed: {file_path}")
                self.callback(f"---")
                metadata["XMP:Status"] = "failed"
                
                if not self.config.dry_run:
                    self.write_metadata(file_path, metadata)
                return
                
            # Send image data to callback for GUI display
            if self.callback and hasattr(self.callback, '__call__'):
                
                # Create a dictionary with image data for GUI
                image_data = {
                    'type': 'image_data',
                    'base64_image': processed_image,
                    'caption': updated_metadata.get('MWG:Description', ''),
                    'keywords': updated_metadata.get('MWG:Keywords', []),
                    'file_path': file_path
                }
                
                self.callback(image_data)    
                
            if not self.config.dry_run:
                self.write_metadata(file_path, updated_metadata)
                
            print(f"{status}: {file_path}")
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
            if status == "success":
                 
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
            print(f"<b>Error processing:</b> {file_path}: {str(e)}")
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
                    data = clean_tags(self.llm_processor.describe_content(task="keywords", processed_image=processed_image))
                else:    
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
                status = "retry"
                            
            else:
                status = "success"
                keywords = self.process_keywords(metadata, keywords)

            new_metadata["MWG:Description"] = caption
            new_metadata["MWG:Keywords"] = keywords
            new_metadata["XMP:Status"] = status
            new_metadata["XMP:Identifier"] = metadata.get("XMP:Identifier", str(uuid.uuid4()))
            new_metadata["SourceFile"] = file_path
            
            return new_metadata
            
        except Exception as e:
            self.callback(f"Parse error for {file_path}: {str(e)}")
            self.callback(f"---")
            metadata["XMP:Status"] = "retry"
            
            return metadata
            
    def write_metadata(self, file_path, metadata):
        """Write metadata using persistent ExifTool instance"""
        if self.config.dry_run:
            print("Dry run. Not writing.")
            
            return True

        try:
            params = ["-P"]
            
            if self.config.no_backup or self.config.use_sidecar:
                params.append("-overwrite_original")
            if self.config.use_sidecar:
                file_path = file_path + ".xmp"
                
            # Use existing ExifTool instance
            self.et.set_tags(file_path, tags=metadata, params=params)
            
            return True
            
        except Exception as e:
            self.callback(f"\nError writing metadata to {file_path}: {str(e)}")
            print(f"\nError writing metadata to {file_path}: {str(e)}")
            self.callback(f"---")
            return False 
    
    def process_keywords(self, metadata, new_keywords):
        """ Normalize extracted keywords and deduplicate them.
            If update is configured, combine the old and new keywords.
        """
        all_keywords = set()
              
        if self.config.update_keywords:
            existing_keywords = metadata.get("MWG:Keywords", [])
            
            if isinstance(existing_keywords, str):
                existing_keywords = existing_keywords.split(",").strip()
                
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
        print(f"An error occurred during processing: {str(e)}")
    
        if callback:
            callback(f"Error: {str(e)}")
            
    finally:
        print("Waiting for indexer to complete...")
        file_processor.indexer.join()
        print("Indexing completed.")
   
if __name__ == "__main__":
    main()
