"""Configuration: defaults, prompts, supported file types, CLI parsing.

Settings come from three layers, each overriding the one before it:

    Config() defaults  ->  --config FILE  ->  command line options

The file is the same JSON the GUI saves, so a job set up in the GUI can be
handed to the CLI unchanged, and a CLI job can be reopened in the GUI. Unknown
keys in the file are kept as attributes rather than rejected, so a settings
file written by a newer version still loads here.
"""
import argparse
import json
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

# Options whose value may be given as "@path/to/file.txt", meaning "read the
# value out of that file". Prompts are long and multi-line, which is a poor fit
# for a shell argument and an awkward one for a JSON string.
TEXT_FILE_OPTIONS = (
    "instruction", "system_instruction", "caption_instruction",
    "tag_instruction", "keyword_instruction",
    "caption_prefill", "keywords_prefill", "caption_and_keywords_prefill",
)

# Options that accept either a list or the GUI's newline/semicolon separated text.
LIST_OPTIONS = ("skip_folders", "banned_words")


def load_settings(path):
    """Read a JSON settings file (the same shape the GUI saves)."""
    with open(os.path.expanduser(path), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of settings")
    return {key: value for key, value in data.items() if value is not None}


def read_text_option(value):
    """Resolve "@file" to that file's contents; anything else is literal."""
    if isinstance(value, str) and value.startswith("@"):
        with open(os.path.expanduser(value[1:]), "r", encoding="utf-8") as fh:
            return fh.read().strip()
    return value


def as_list(value):
    """Accept a real list, or newline/semicolon separated text from the GUI."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    items = []
    for line in str(value).replace(";", "\n").splitlines():
        if line.strip():
            items.append(line.strip())
    return items


class Config:
    def __init__(self):
        self.directory = None
        # A bare `llmii DIR` should reach a local KoboldCpp without further
        # argument; the GUI overwrites both from its own fields.
        self.api_url = "http://localhost:5001"
        self.api_password = ""
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
        # Most keywords one response may share a leading word before the rest
        # are treated as a prefix-locked run and trimmed. 0 disables.
        self.max_shared_leaders = 5
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
        # Assistant-turn prefill: primes the start of the response so a model
        # that likes to preface its answer with chatter never gets the chance.
        # Ignored whenever use_json_grammar applies to the task, since grammar
        # and prefill can't be combined (see llm_client.py).
        self.use_prefill = False
        self.caption_prefill = "A "
        self.keywords_prefill = '{"Keywords": ["'
        self.caption_and_keywords_prefill = '{"Description": "A '
        self.skip_folders = []
        self.rename_invalid = False
        self.preserve_date = False
        self.fix_extension = False
        self.banned_words = []
        self.chunk_size = 100
        # Number of images to pre-decode while the LLM works on the current
        # one. 1 overlaps CPU/disk with the network round trip; 0 disables.
        self.prefetch = 1
        self.json_output = False
        self.json_output_file = "image_tags.json"

        self.instruction = DEFAULT_INSTRUCTION
        self.image_extensions = IMAGE_EXTENSIONS

    # ------------------------------------------------------------------
    # layering
    # ------------------------------------------------------------------
    def apply(self, settings):
        """Layer a dict of settings over this config, resolving "@file" values."""
        for key, value in settings.items():
            if key in TEXT_FILE_OPTIONS:
                value = read_text_option(value)
            setattr(self, key, value)
        return self

    def normalize(self):
        """Coerce the free-form options into the shapes the pipeline expects."""
        for key in LIST_OPTIONS:
            setattr(self, key, as_list(getattr(self, key, None)))
        return self

    @property
    def prefills(self):
        """Per-task prefill strings, in the shape LLMProcessor expects."""
        return {
            "caption": self.caption_prefill,
            "keywords": self.keywords_prefill,
            "caption_and_keywords": self.caption_and_keywords_prefill,
        }

    def to_dict(self):
        """The settings as JSON-ready data, for saving a job to a file."""
        return {key: value for key, value in vars(self).items()
                if key != "image_extensions"}

    # ------------------------------------------------------------------
    # command line
    # ------------------------------------------------------------------
    @classmethod
    def build_parser(cls, defaults=None):
        """Build the CLI parser.

        Options carry no argparse defaults (`SUPPRESS`), so an option that is
        not given simply does not appear in the parsed result and cannot
        overwrite a value from --config. Help text shows the real default from
        Config, which is also the value used when nothing sets it.
        """
        base = defaults or cls()

        def shown(attribute):
            value = getattr(base, attribute)
            return f" (default: {value if value != '' else 'empty'})"

        parser = argparse.ArgumentParser(
            prog="llmii",
            description="Label images with a local vision model.",
            argument_default=argparse.SUPPRESS,
        )
        parser.add_argument("directory", nargs="?",
                            help="Directory containing the files")
        parser.add_argument("--config", metavar="FILE",
                            help="JSON settings file, in the same format the GUI "
                                 "saves; command line options override it")

        api = parser.add_argument_group("API")
        api.add_argument("--api-url", help="URL for the LLM API" + shown("api_url"))
        api.add_argument("--api-password", help="Password for the LLM API")
        api.add_argument("--gen-count", type=int,
                         help="Number of tokens to generate" + shown("gen_count"))
        api.add_argument("--res-limit", type=int,
                         help="Limit the resolution of the image" + shown("res_limit"))

        prompts = parser.add_argument_group(
            "instructions",
            "Each accepts literal text, or @FILE to read the text from a file.")
        prompts.add_argument("--instruction",
                             help="Main instruction, used for caption and keywords "
                                  "in one generation")
        prompts.add_argument("--system-instruction", help="System prompt")
        prompts.add_argument("--caption-instruction",
                             help="Instruction for the detailed caption generation")
        prompts.add_argument("--tag-instruction",
                             help="Instruction for the keywords-only generation")

        prefill = parser.add_argument_group(
            "prefill",
            "Prime the start of the model's response, so it can't preface the "
            "answer with commentary. Ignored for a task where --json-grammar "
            "applies. Each accepts literal text, or @FILE to read it from a file.")
        prefill.add_argument("--use-prefill", action="store_true",
                             help="Enable response prefill" + shown("use_prefill"))
        prefill.add_argument("--no-use-prefill", dest="use_prefill",
                             action="store_false", help=argparse.SUPPRESS)
        prefill.add_argument("--caption-prefill",
                             help="Prefill for the detailed caption generation"
                                  + shown("caption_prefill"))
        prefill.add_argument("--keywords-prefill",
                             help="Prefill for the keywords-only generation"
                                  + shown("keywords_prefill"))
        prefill.add_argument("--caption-and-keywords-prefill",
                             help="Prefill for the combined caption+keywords "
                                  "generation" + shown("caption_and_keywords_prefill"))

        crawl = parser.add_argument_group("files")
        crawl.add_argument("--no-crawl", action="store_true",
                           help="Disable recursive indexing")
        crawl.add_argument("--skip-folders",
                           help="Folder names to skip, separated by semicolons")
        crawl.add_argument("--no-backup", action="store_true",
                           help="Don't make a backup of files before writing")
        crawl.add_argument("--dry-run", action="store_true",
                           help="Don't write any files")
        crawl.add_argument("--rename-invalid", action="store_true",
                           help="Rename invalid files so they don't get reprocessed")
        crawl.add_argument("--preserve-date", action="store_true",
                           help="Keep the original modified date (uses a temp file "
                                "when writing)")
        crawl.add_argument("--fix-extension", action="store_true",
                           help="Correct file extensions that don't match the content")
        crawl.add_argument("--skip-verify", action="store_true",
                           help="Skip verifying file metadata validity before processing")
        crawl.add_argument("--chunk-size", type=int,
                           help="Files indexed per batch" + shown("chunk_size"))
        crawl.add_argument("--prefetch", type=int,
                           help="Images decoded ahead of the model; 0 disables"
                                + shown("prefetch"))

        rerun = parser.add_argument_group("reprocessing")
        rerun.add_argument("--reprocess-all", action="store_true",
                           help="Reprocess all files")
        rerun.add_argument("--reprocess-failed", action="store_true",
                           help="Reprocess failed files")
        rerun.add_argument("--reprocess-orphans", action="store_true",
                           help="If a file has a UUID, determine its status")
        rerun.add_argument("--quick-fail", action="store_true",
                           help="Mark failed after one try")

        out = parser.add_argument_group("output")
        out.add_argument("--update-keywords", action="store_true",
                         help="Update existing keyword metadata")
        out.add_argument("--update-caption", action="store_true",
                         help="Add the generated caption to the existing description tag")
        out.add_argument("--detailed-caption", action="store_true",
                         help="Write a detailed caption along with keywords "
                              "(two generations)")
        out.add_argument("--short-caption", action="store_true",
                         help="Write a caption and keywords in one generation")
        out.add_argument("--no-caption", action="store_true",
                         help="Do not modify caption")
        out.add_argument("--use-sidecar", action="store_true",
                         help="Store generated data in an xmp sidecar instead of "
                              "the image file")
        out.add_argument("--no-sidecar-extension", action="store_true",
                         help="Does not add the image file extension to sidecar filenames")
        out.add_argument("--json-output", action="store_true",
                         help="Write results to a JSON file instead of image "
                              "metadata; descriptions are included when a caption "
                              "option is also given")
        out.add_argument("--json-output-file",
                         help="Path for the JSON output file"
                              + shown("json_output_file"))

        words = parser.add_argument_group("keywords")
        words.add_argument("--banned-words",
                           help="Words to drop from generated keywords, separated "
                                "by semicolons, or @FILE for one per line")
        words.add_argument("--max-shared-leaders", type=int,
                           help="Trim keywords past the Nth sharing a leading word "
                                "(catches prefix-locked generations; 0 disables)"
                                + shown("max_shared_leaders"))
        words.add_argument("--max-words-per-keyword", type=int,
                           help="Words allowed in one keyword"
                                + shown("max_words_per_keyword"))
        for name, dest, description in (
                ("normalize-keywords", "normalize_keywords",
                 "keyword normalization"),
                ("depluralize-keywords", "depluralize_keywords",
                 "converting plural keywords to singular"),
                ("limit-word-count", "limit_word_count",
                 "the limit on words per keyword"),
                ("split-and-entries", "split_and_entries",
                 "splitting 'and'/'or' keywords into separate entries"),
                ("ban-prompt-words", "ban_prompt_words",
                 "dropping words echoed back from the prompt"),
                ("no-digits-start", "no_digits_start",
                 "dropping keywords that start with 3+ digits"),
                ("min-word-length", "min_word_length",
                 "dropping one-character words"),
                ("latin-only", "latin_only",
                 "dropping keywords with non-Latin characters"),
        ):
            state = "on" if getattr(base, dest) else "off"
            words.add_argument(f"--{name}", dest=dest, action="store_true",
                               help=f"Enable {description} (default: {state})")
            words.add_argument(f"--no-{name}", dest=dest, action="store_false",
                               help=argparse.SUPPRESS)

        sampler = parser.add_argument_group("samplers")
        sampler.add_argument("--temperature", type=float,
                             help="Randomness of the output" + shown("temperature"))
        sampler.add_argument("--top-p", type=float, help="Top-p" + shown("top_p"))
        sampler.add_argument("--top-k", type=int, help="Top-k" + shown("top_k"))
        sampler.add_argument("--min-p", type=float, help="Min-p" + shown("min_p"))
        sampler.add_argument("--rep-pen", type=float,
                             help="Repetition penalty" + shown("rep_pen"))
        sampler.add_argument("--json-grammar", dest="use_json_grammar",
                             action="store_true",
                             help="Constrain output to the expected JSON schema"
                                  + shown("use_json_grammar"))
        sampler.add_argument("--no-json-grammar", dest="use_json_grammar",
                             action="store_false", help=argparse.SUPPRESS)

        return parser

    @classmethod
    def from_args(cls, argv=None):
        config = cls()
        parser = cls.build_parser(config)
        args = vars(parser.parse_args(argv))

        settings_file = args.pop("config", None)
        if settings_file:
            config.apply(load_settings(settings_file))
        config.apply(args)  # an explicit option always wins over the file
        config.normalize()

        if not config.directory:
            parser.error('no directory given: pass one, or set "directory" '
                         'in the --config file')
        return config
