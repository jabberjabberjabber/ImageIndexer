"""Configuration: defaults, prompts, supported file types, CLI parsing."""
import argparse
import os

# Get project root directory 
PROJECT_ROOT = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Resources directory at project root level
RESOURCES_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "resources"))

DEFAULT_INSTRUCTION = """Return a JSON object containing a Description for the image and a list of Keywords.

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

IMAGE_EXTENSIONS = {
    "JPEG": [".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi",
             ".jp2", ".j2k", ".jpf", ".jpx", ".jpm", ".mj2"],
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
    ],
}


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
        self.res_limit = 768
        self.detailed_caption = False
        self.short_caption = False
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
        self.tag_instruction = (
            'Return a JSON object with key Keywords with the value as array of '
            'Keywords and tags that describe the image as follows: {"Keywords": []}'
        )
        self.no_sidecar_extension = False
        # Sampler settings
        self.temperature = 0.1
        self.top_p = 0.8
        self.rep_pen = 1.00
        self.top_k = 100
        self.min_p = 0.0
        self.use_default_badwordsids = False
        self.use_json_grammar = False
        self.skip_folders = []
        self.rename_invalid = False
        self.preserve_date = False
        self.fix_extension = False
        self.banned_words = []
        self.chunk_size = 100
        # Number of images to pre-decode while the LLM works on the current
        # one. 1 overlaps CPU/disk with the network round trip; 0 disables.
        self.prefetch = 1

        self.instruction = DEFAULT_INSTRUCTION
        self.image_extensions = IMAGE_EXTENSIONS

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Image Indexer")
        parser.add_argument("directory", help="Directory containing the files")
        parser.add_argument("--api-url", default="http://localhost:5001",
                            help="URL for the LLM API")
        parser.add_argument("--api-password", default="",
                            help="Password for the LLM API")
        parser.add_argument("--gen-count", type=int, default=150,
                            help="Number of tokens to generate")
        parser.add_argument("--res-limit", type=int, default=448,
                            help="Limit the resolution of the image")
        parser.add_argument("--no-crawl", action="store_true",
                            help="Disable recursive indexing")
        parser.add_argument("--no-backup", action="store_true",
                            help="Don't make a backup of files before writing")
        parser.add_argument("--dry-run", action="store_true",
                            help="Don't write any files")
        parser.add_argument("--reprocess-all", action="store_true",
                            help="Reprocess all files")
        parser.add_argument("--reprocess-failed", action="store_true",
                            help="Reprocess failed files")
        parser.add_argument("--reprocess-orphans", action="store_true",
                            help="If a file has a UUID, determine its status")
        parser.add_argument("--update-keywords", action="store_true",
                            help="Update existing keyword metadata")
        parser.add_argument("--update-caption", action="store_true",
                            help="Add the generated caption to the existing description tag")
        parser.add_argument("--detailed-caption", action="store_true",
                            help="Write a detailed caption along with keywords")
        parser.add_argument("--short-caption", action="store_true",
                            help="Write a short caption along with keywords")
        parser.add_argument("--no-caption", action="store_true",
                            help="Do not modify caption")
        parser.add_argument("--use-sidecar", action="store_true",
                            help="Store generated data in an xmp sidecar instead of the image file")
        parser.add_argument("--no-sidecar-extension", action="store_true",
                            help="Does not add the image file extension to sidecar filenames")
        parser.add_argument("--skip-verify", action="store_true",
                            help="Skip verifying file metadata validity before processing")
        parser.add_argument("--quick-fail", action="store_true",
                            help="Mark failed after one try")
        parser.add_argument("--normalize-keywords", action="store_true",
                            help="Enable keyword normalization")
        parser.add_argument("--rename-invalid", action="store_true",
                            help="Rename invalid files so they don't get reprocessed")
        parser.add_argument("--preserve-date", action="store_true",
                            help="Keep the original modified date (uses a temp file when writing)")
        args = parser.parse_args()

        config = cls()
        defaults = {a.dest: parser.get_default(a.dest) for a in parser._actions}
        for key, value in vars(args).items():
            if value != defaults.get(key) or not hasattr(config, key):
                setattr(config, key, value)
        return config
