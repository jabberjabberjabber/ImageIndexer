"""Entry point and backward-compatible facade.

The implementation now lives in focused modules:

    config.py       - Config, prompts, supported extensions, CLI parsing
    keywords.py     - keyword normalization (pure functions)
    llm_output.py   - parsing/repair of raw LLM responses
    llm_client.py   - HTTP client for the vision LLM (LLMProcessor)
    indexer.py      - background directory crawler (BackgroundIndexer)
    metadata_io.py  - ExifTool reads/writes, sidecars, file repair ops
    pipeline.py     - FileProcessor orchestration with image prefetch

Everything that was importable from this module before still is.
"""
from .config import Config
from .keywords import normalize_keyword, split_on_internal_capital
from .llm_output import clean_json, clean_string, clean_tags, markdown_list_to_dict
from .llm_client import LLMProcessor
from .indexer import BackgroundIndexer
from .pipeline import FileProcessor

__all__ = [
    "Config", "normalize_keyword", "split_on_internal_capital",
    "clean_json", "clean_string", "clean_tags", "markdown_list_to_dict",
    "LLMProcessor", "BackgroundIndexer", "FileProcessor", "main",
]


def main(config=None, callback=None, check_paused_or_stopped=None):
    if config is None:
        config = Config.from_args()

    if not hasattr(config, "chunk_size"):
        config.chunk_size = 100

    file_processor = FileProcessor(config, check_paused_or_stopped, callback)

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
