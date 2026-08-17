import os
import queue
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .image_processor import ImageProcessor
from .indexer import BackgroundIndexer
from .keywords import limit_shared_leaders, normalize_keyword
from .llm_client import LLMProcessor
from .llm_output import clean_json, clean_string, clean_tags
from . import metadata_io
from .metadata_io import (
    MetadataStore, KEYWORD_FIELDS, CAPTION_FIELDS,
    IDENTIFIER_FIELDS, STATUS_FIELDS, ORIENTATION_FIELD,
)


class FileProcessor:
    def __init__(self, config, check_paused_or_stopped=None, callback=None):
        self.config = config
        self.check_paused_or_stopped = check_paused_or_stopped or (lambda: False)
        self.callback = callback or print

        self.llm_processor = LLMProcessor(config)
        self.image_processor = ImageProcessor(
            max_dimension=config.res_limit, patch_sizes=[14])
        self.store = MetadataStore(config, self.callback)

        self.banned_words = frozenset(w.lower() for w in config.banned_words)

        self._ext_to_type = {
            ext.lower(): ftype
            for ftype, exts in config.image_extensions.items()
            for ext in exts
        }

        self.failed_validations = []
        self.total_processing_time = 0
        self.files_processed = 0
        self.files_completed = 0

        self.metadata_queue = queue.Queue()
        self.indexer = BackgroundIndexer(
            config.directory,
            self.metadata_queue,
            list(self._ext_to_type.keys()),
            config.no_crawl,
            chunk_size=getattr(config, "chunk_size", 100),
            skip_folders=getattr(config, "skip_folders", []),
        )
        self.indexer.start()

    def process_directory(self, directory):
        prefetch = max(0, getattr(self.config, "prefetch", 1))
        executor = ThreadPoolExecutor(max_workers=max(1, prefetch)) if prefetch else None

        try:
            while not (self.indexer.indexing_complete and self.metadata_queue.empty()):
                if self.check_pause_stop():
                    return
                try:
                    directory, files = self.metadata_queue.get(timeout=1)
                except queue.Empty:
                    continue

                self.callback(f"Processing directory: {directory}")
                self.callback("---")

                batch_size = 50
                for i in range(0, len(files), batch_size):
                    batch = files[i:i + batch_size]
                    if batch:
                        print(f"Reading metadata for {len(batch)} file(s)...")
                    metadata_list = self.store.get_batch(batch)

                    work = []
                    for raw in metadata_list:
                        if not raw:
                            continue
                        metadata = self._standardize(raw)
                        metadata = self._triage(metadata)
                        if metadata:
                            work.append(metadata)
                        else:
                            self.files_processed += 1
                        if self.check_pause_stop():
                            return

                    if self._run_pipeline(work, executor):
                        return  # stopped

                self.update_progress()
        finally:
            if executor:
                executor.shutdown(wait=False)
            try:
                self.store.terminate()
                self.callback("ExifTool process terminated cleanly")
            except Exception as e:
                self.callback(f"Warning: ExifTool termination error: {str(e)}")

    def _run_pipeline(self, work, executor):
        """Run preprocess+inference over triaged items, prefetching the next
        image while the LLM handles the current one. Returns True if a
        pause/stop request should abort processing.
        """
        
        if not work:
            return False

        if executor is None:
            for metadata in work:
                self._process_counted(metadata, *self._preprocess(metadata))
                if self.check_pause_stop():
                    return True
            return False

        depth = max(1, getattr(self.config, "prefetch", 1))
        pending = deque()
        it = iter(work)

        def submit_next():
            try:
                m = next(it)
            except StopIteration:
                return
            pending.append((m, executor.submit(self._preprocess, m)))

        for _ in range(depth + 1):
            submit_next()

        while pending:
            metadata, future = pending.popleft()
            processed_image, error = future.result()
            submit_next()
            self._process_counted(metadata, processed_image, error)
            if self.check_pause_stop():
                for _m, f in pending:
                    f.cancel()
                return True
        return False

    def _process_counted(self, metadata, processed_image, error):
        """_process_one, with the queue counter advanced for this one file.

        Counted before the call, not after, so the "Processed" line a file
        prints when it finishes includes itself. Both pipelines route through
        here; _process_one is overridden, this is not.
        """
        self.files_processed += 1
        self._process_one(metadata, processed_image, error)

    def _standardize(self, metadata):
        """Collapse the many possible tag names into MWG/XMP fields.
        """
        
        # If the read target was a sidecar, point SourceFile back at the image
        source = metadata["SourceFile"]
        if source.lower().endswith(".xmp"):
            image = self.store.image_for_read_target(source)
            if image:
                metadata["SourceFile"] = image

        new_metadata = {"SourceFile": metadata.get("SourceFile")}

        keywords = []
        caption = status = identifier = None
        filetype = filetype_ext = None
        orientation = 1

        for key, value in metadata.items():
            if key in KEYWORD_FIELDS:
                if isinstance(value, list):
                    keywords.extend(value)
                else:
                    keywords.append(value)
            elif key in CAPTION_FIELDS:
                if caption is None:
                    caption = value
            elif key in IDENTIFIER_FIELDS:
                identifier = value
            elif key in STATUS_FIELDS:
                status = value
            elif key == "File:FileType":
                filetype = value
            elif key == "File:FileTypeExtension":
                filetype_ext = value
            elif key == ORIENTATION_FIELD:
                orientation = value

        if keywords:
            new_metadata["MWG:Keywords"] = keywords
        if caption:
            new_metadata["MWG:Description"] = caption
        if status:
            new_metadata["XMP:Status"] = status
        if identifier:
            new_metadata["XMP:Identifier"] = identifier
        if not self.config.skip_verify and "ExifTool:Validate" in metadata:
            new_metadata["ExifTool:Validate"] = metadata["ExifTool:Validate"]
        if filetype:
            new_metadata["File:FileType"] = filetype
        if filetype_ext:
            new_metadata["File:FileTypeExtension"] = filetype_ext
        if orientation:
            new_metadata["EXIF:Orientation"] = orientation

        return new_metadata

    def _triage(self, metadata):
        """Decide whether this file needs LLM processing. Returns metadata
        ready for processing, or None to skip.
        """
        
        try:
            file_path = metadata["SourceFile"]

            if not os.path.exists(file_path):
                self.callback(f"File no longer exists: {file_path}")
                self.callback("---")
                return None

            current_status = metadata.get("XMP:Status")
            if current_status == "invalid" and not self.config.reprocess_all:
                self.callback(f"Skipping file marked as invalid: {file_path}")
                self.callback("---")
                return None

            should_validate = ((not current_status or self.config.reprocess_all)
                               and not self.config.skip_verify)
            if should_validate and not self._validate(metadata, file_path):
                return None

            metadata = self.check_uuid(metadata, file_path)
            if not metadata:
                return None

            if self.get_file_type(os.path.splitext(file_path)[1]) is None:
                self.callback(f"Not a supported image type: {file_path}")
                self.callback("---")
                return None

            return metadata

        except Exception as e:
            print(f"Triage Error: {type(e).__name__}: {str(e)}")
            return None

    def _validate(self, metadata, file_path):
        """Returns False if the file is invalid or unwritable.
        """
        
        validation_parts = metadata.get("ExifTool:Validate", "0 0 0").split()
        if len(validation_parts) >= 3:
            errors, warnings, minor = map(int, validation_parts[:3])
        else:
            errors, warnings, minor = 0, 0, 0

        if errors > 0:
            print(f"Validation Failed: {os.path.basename(file_path)}")
            print(f"  Errors: {errors}, Warnings: {warnings}, Minor: {minor}")
            self.callback(f"\nValidation failed: {file_path}")
            self.callback(f"  Errors: {errors}, Warnings: {warnings}, Minor: {minor}")
            self.failed_validations.append(file_path)
            if self.config.rename_invalid:
                metadata_io.rename_to_invalid(file_path, self.callback)
            self.callback("---")
            return False

        # Test writability
        if warnings > 0 and minor >= warnings:
            print(f"File has validation warnings: {os.path.basename(file_path)}")
            print(f"  Warnings: {warnings}, Minor: {minor} - Testing writeability...")
            test_metadata = {"SourceFile": file_path, "XMP:Status": "valid"}
            if not self.write_metadata(file_path, test_metadata):
                print("  Metadata cannot be written to file")
                self.callback(f"\nMetadata is not writable: {file_path}")
                self.failed_validations.append(file_path)
                self.callback("---")
                return False
            print("  File is writable - proceeding")
            metadata["XMP:Status"] = "valid"

        return True

    def check_uuid(self, metadata, file_path):
        """Very important.
        """
        try:
            status = metadata.get("XMP:Status")
            identifier = metadata.get("XMP:Identifier")
            keywords = metadata.get("MWG:Keywords")

            # Orphan: has UUID and keywords but no status
            if (identifier and self.config.reprocess_orphans
                    and keywords and not status):
                metadata["XMP:Status"] = "success"
                try:
                    if self.store.write(file_path, metadata,
                                        on_write_error=self._on_write_error):
                        print(f"Status added for orphan: {file_path}")
                        self.callback(f"Status added for orphan: {file_path}")
                        if not self.config.reprocess_all:
                            return None
                    else:
                        print(f"Metadata write error for orphan: {file_path}")
                        self.callback(f"Metadata write error for orphan: {file_path}")
                        return None
                except Exception:
                    print("Error writing orphan status")
                    return None
                status = "success"

            if identifier:
                if not self.config.reprocess_all and status == "success":
                    return None
                if self.config.reprocess_all or status == "retry":
                    metadata["XMP:Status"] = None
                    return metadata
                if status == "failed":
                    if self.config.reprocess_failed or self.config.reprocess_all:
                        metadata["XMP:Status"] = None
                        return metadata
                    return None
                if not keywords:
                    metadata["XMP:Status"] = None
                    return metadata
                return None

            # No UUID: new file
            metadata["XMP:Identifier"] = str(uuid.uuid4())
            return metadata

        except Exception as e:
            print(f"Error checking UUID: {str(e)}")
            return None

    def _preprocess(self, metadata):
        """Decode/resize/encode the image. Returns (base64_or_None, error).
        """
        
        try:
            orientation = metadata.get("EXIF:Orientation") or 1
            processed_image, _path = self.image_processor.process_image(
                metadata["SourceFile"], orientation)
            return processed_image, None
        except Exception as e:
            return None, e

    def _process_one(self, metadata, processed_image, error=None):
        """Run LLM generation for one triaged, preprocessed file and write
        the result.
        """
        
        try:
            file_path = metadata["SourceFile"]

            if error is not None:
                print(f"Image Processing Error: {os.path.basename(file_path)}")
                print(f"  Error type: {type(error).__name__}")
                print(f"  Details: {str(error)}")
                self.callback(f"Image processing error for {file_path}: {str(error)}")
                if self.config.rename_invalid:
                    metadata_io.rename_to_invalid(file_path, self.callback)
                self.callback("---")
                return

            if not processed_image:
                print(f"Image Processing Failed: {os.path.basename(file_path)}")
                print("  Could not generate base64 image data")
                self.callback(f"Failed to process image: {file_path}")
                if self.config.rename_invalid:
                    metadata_io.rename_to_invalid(file_path, self.callback)
                self.callback("---")
                return

            filetype = metadata.get("File:FileType",
                                    self.get_file_type(os.path.splitext(file_path)[1]))
            print(f"Processing: {os.path.basename(file_path)} [{filetype}]")

            success = True
            start_time = time.time()

            updated_metadata = self.generate_metadata(metadata, processed_image)
            status = updated_metadata.get("XMP:Status")

            # Retry once on failure
            if not self.config.quick_fail and status == "retry":
                print(f"AI Generation Issue - Retrying: {os.path.basename(file_path)}")
                print("  Reason: No valid keywords generated on first attempt")
                self.callback(f"Asking AI to try again for {file_path}...")
                self.callback("---")
                updated_metadata = self.generate_metadata(metadata, processed_image)
                status = updated_metadata.get("XMP:Status")

            if status != "success":
                print(f"AI Generation Failed: {os.path.basename(file_path)}")
                print("  The AI could not generate valid keywords after retry")
                self.callback(f"Retry failed due to AI for {file_path}")
                self.callback("---")
                metadata["XMP:Status"] = "failed"
                if not self.config.dry_run:
                    success = False
                    self.write_metadata(file_path, {
                        "SourceFile": file_path,
                        "XMP:Status": "failed",
                        "XMP:Identifier": metadata.get("XMP:Identifier",
                                                       str(uuid.uuid4())),
                    })

            # Fix file extension if enabled (before writing metadata)
            if self.config.fix_extension and success:
                expected_ext = metadata.get("File:FileTypeExtension")
                if expected_ext:
                    new_file_path = metadata_io.fix_file_extension(
                        file_path, expected_ext, self.callback)
                    if new_file_path != file_path:
                        file_path = new_file_path
                        updated_metadata["SourceFile"] = file_path

            # Send image data to callback for GUI display
            if callable(self.callback) and success:
                self.callback({
                    "type": "image_data",
                    "base64_image": processed_image,
                    "caption": updated_metadata.get("MWG:Description", ""),
                    "keywords": updated_metadata.get("MWG:Keywords", []),
                    "file_path": file_path,
                })

            if not self.config.dry_run and success:
                if self.store.write(file_path, updated_metadata,
                                    on_write_error=self._on_write_error):
                    print("  Metadata written successfully")
                else:
                    success = False

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.files_completed += 1
            self._report(file_path, status, processing_time, success)

        except Exception as e:
            file_path = metadata.get("SourceFile", "?")
            print(f"Processing Error: {os.path.basename(file_path)}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Details: {str(e)}")
            self.callback(f"<b>Error processing:</b> {file_path}: {str(e)}")
            self.callback("---")

    def _report(self, file_path, status, processing_time, success):
        in_queue = max(0, self.indexer.total_files_found - self.files_processed)
        average_time = self.total_processing_time / self.files_completed
        time_left = max(0, average_time * in_queue)
        time_left_unit = "s"
        if time_left > 180:
            time_left /= 60
            time_left_unit = "mins"

        if success:
            self.callback(f"<b>Image:</b> {os.path.basename(file_path)}")
            self.callback(f"<b>Status:</b> {status}")
            self.callback(
                f"<b>Processing time:</b> {processing_time:.2f}s, "
                f"<b>Average processing time:</b> {average_time:.2f}s"
            )
            self.callback(
                f"<b>Processed:</b> {self.files_processed}, "
                f"<b>In queue:</b> {in_queue}, "
                f"<b>Time remaining (est):</b> {time_left:.2f}{time_left_unit}"
            )
            self.callback("---")

    def generate_metadata(self, metadata, processed_image):
        """Generate metadata

        detailed_caption: keywords + a detailed caption (two generations).
        short_caption: caption and keywords in one generation.
        update_caption: appends the new caption to the existing description.

        Returns a metadata dict.
        """
        new_metadata = {}
        existing_caption = metadata.get("MWG:Description")
        caption = None
        keywords = None
        file_path = metadata["SourceFile"]

        try:
            if not self.config.no_caption and self.config.detailed_caption:
                print("  Generating keywords and detailed caption...")
                data = clean_tags(self.llm_processor.describe_content(
                    task="keywords", processed_image=processed_image))
                detailed_caption = clean_string(self.llm_processor.describe_content(
                    task="caption", processed_image=processed_image)) or ""

                if existing_caption and self.config.update_caption:
                    caption = (existing_caption + "<generated>"
                               + detailed_caption + "</generated>")
                else:
                    caption = detailed_caption

                if isinstance(data, dict):
                    keywords = data.get("Keywords")

            else:
                if self.config.no_caption or not self.config.short_caption:
                    print("  Generating keywords only...")
                    data = clean_tags(self.llm_processor.describe_content(
                        task="keywords", processed_image=processed_image))
                else:
                    print("  Generating caption and keywords...")
                    data = clean_json(self.llm_processor.describe_content(
                        task="caption_and_keywords", processed_image=processed_image))

                if isinstance(data, dict):
                    keywords = data.get("Keywords")
                    description = data.get("Description") or ""

                    if existing_caption and self.config.update_caption:
                        caption = (existing_caption + "<generated>"
                                   + description + "</generated>")
                    elif description and not self.config.no_caption:
                        caption = description
                    else:
                        caption = existing_caption or ""

            if not keywords:
                print(f"No Keywords Generated: {os.path.basename(file_path)}")
                print("  AI response did not contain valid keywords")
                status = "retry"
            else:
                status = "success"
                keywords = self.process_keywords(metadata, keywords)
                if keywords:
                    print(f"Generated {len(keywords)} keyword(s) for: "
                          f"{os.path.basename(file_path)}")

            new_metadata["MWG:Description"] = caption
            new_metadata["MWG:Keywords"] = keywords
            new_metadata["XMP:Status"] = status
            new_metadata["XMP:Identifier"] = metadata.get(
                "XMP:Identifier", str(uuid.uuid4()))
            new_metadata["SourceFile"] = file_path
            return new_metadata

        except Exception as e:
            print(f"Metadata Generation Error: {os.path.basename(file_path)}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Details: {str(e)}")
            self.callback(f"Parse error for {file_path}: {str(e)}")
            self.callback("---")
            return {
                "MWG:Description": existing_caption,
                "MWG:Keywords": metadata.get("MWG:Keywords", []),
                "XMP:Status": "retry",
                "XMP:Identifier": metadata.get("XMP:Identifier", str(uuid.uuid4())),
                "SourceFile": file_path,
            }

    def process_keywords(self, metadata, new_keywords):
        """
        Normalize keywords, dedupe, and merge old ones if configured.
        Generation order is preserved. 
        """
        
        # dict keys: dedupe with insertion order preserved
        generated = {}

        def add(bucket, keyword):
            normalized = normalize_keyword(keyword, self.banned_words, self.config)
            if isinstance(normalized, list):
                for part in normalized:
                    bucket.setdefault(part, None)
            elif normalized:
                bucket.setdefault(normalized, None)

        for kw in new_keywords:
            add(generated, kw)

        kept, dropped = limit_shared_leaders(
            list(generated), getattr(self.config, "max_shared_leaders", 0)
        )
        if dropped:
            leader = dropped[0].split()[0]
            print(f"  Trimmed {len(dropped)} prefix-locked keywords ('{leader} ...')")
            self.callback(
                f"Trimmed {len(dropped)} repetitive keywords starting with '{leader}'"
            )

        all_keywords = dict.fromkeys(kept)

        if self.config.update_keywords:
            existing = metadata.get("MWG:Keywords", [])
            if isinstance(existing, str):
                existing = [k.strip() for k in existing.split(",")]
            for kw in existing:
                add(all_keywords, kw)

        return list(all_keywords) if all_keywords else None

    def write_metadata(self, file_path, metadata):
        return self.store.write(file_path, metadata,
                                on_write_error=self._on_write_error)

    def _on_write_error(self, image_path, write_target, writing_sidecar):
        if self.config.rename_invalid:
            metadata_io.rename_to_invalid(image_path, self.callback)
            
            # Only ever delete a sidecar -- never the image.
            if (writing_sidecar and write_target != image_path
                    and os.path.exists(write_target)):
                try:
                    os.remove(write_target)
                    self.callback("Removed incomplete sidecar file")
                except OSError:
                    pass

    def get_file_type(self, file_ext):
        """Return the type key for a supported extension (.nef -> RAW),
        else None.
        """
            
        if not file_ext.startswith("."):
            file_ext = "." + file_ext
        return self._ext_to_type.get(file_ext.lower())

    def check_pause_stop(self):
        if self.check_paused_or_stopped():
            while self.check_paused_or_stopped():
                time.sleep(0.1)
            if self.check_paused_or_stopped():
                return True
        return False

    def update_progress(self):
        files_remaining = max(0, self.indexer.total_files_found - self.files_processed)
        self.callback(f"Directory processed. Files remaining in queue: {files_remaining}")
        self.callback("---")
