from .config import Config
from .keywords import normalize_keyword, split_on_internal_capital
from .llm_output import clean_json, clean_string, clean_tags, markdown_list_to_dict
from .llm_client import LLMProcessor
from .indexer import BackgroundIndexer
from .pipeline import FileProcessor
from .json_pipeline import JsonFileProcessor

__all__ = [
    "Config", "normalize_keyword", "split_on_internal_capital",
    "clean_json", "clean_string", "clean_tags", "markdown_list_to_dict",
    "LLMProcessor", "BackgroundIndexer", "FileProcessor", "JsonFileProcessor",
    "main",
]


def main(config=None, callback=None, check_paused_or_stopped=None):
    if config is None:
        config = Config.from_args()

    if not hasattr(config, "chunk_size"):
        config.chunk_size = 100

    if getattr(config, "json_output", False):
        file_processor = JsonFileProcessor(config, check_paused_or_stopped, callback)
    else:
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
