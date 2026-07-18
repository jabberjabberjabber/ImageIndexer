"""Background directory indexer.

Runs in a thread, feeding (directory, [files]) batches into a queue so the
pipeline can start on the first batch while the rest of the tree is still
being scanned.

"""
import os
import queue
import threading


class BackgroundIndexer(threading.Thread):
    def __init__(self, root_dir, metadata_queue, file_extensions,
                 no_crawl=False, chunk_size=100, skip_folders=None):
        super().__init__(daemon=True)
        self.root_dir = root_dir
        self.metadata_queue = metadata_queue
        self.extensions = frozenset(ext.lower() for ext in file_extensions)
        self.no_crawl = no_crawl
        self.skip_folders = skip_folders or []
        self._skip_norms = {os.path.normpath(s) for s in self.skip_folders}
        self._skip_abs = {os.path.normpath(os.path.join(root_dir, s))
                          for s in self.skip_folders}
        self.total_files_found = 0
        self.indexing_complete = False
        self.chunk_size = chunk_size

    def _should_skip_directory(self, directory):
        if not self.skip_folders:
            return False
        dir_norm = os.path.normpath(directory)
        if dir_norm in self._skip_norms or dir_norm in self._skip_abs:
            return True
        parts = set(dir_norm.split(os.sep))
        return not self._skip_norms.isdisjoint(parts)

    def run(self):
        if self.no_crawl:
            if not self._should_skip_directory(self.root_dir):
                print(f"Indexing directory (no crawl): {self.root_dir}")
                self._index_directory(self.root_dir)
        else:
            directories = []
            for root, dirnames, _ in os.walk(self.root_dir):
                # Prune skipped subtrees
                dirnames[:] = [d for d in dirnames
                               if not self._should_skip_directory(os.path.join(root, d))]
                dir_path = os.path.normpath(root)
                if not self._should_skip_directory(dir_path):
                    directories.append(dir_path)

            directories.sort()
            print(f"Found {len(directories)} director(ies) to index")
            for directory in directories:
                self._index_directory(directory)

        print(f"Indexing complete. Total files found: {self.total_files_found}")
        self.indexing_complete = True

    def _flush(self, directory, batch):
        self.total_files_found += len(batch)
        self.metadata_queue.put((directory, batch))

    def _index_directory(self, directory):
        """Scan one directory, emitting file batches of chunk_size."""
        directory = os.path.normpath(directory)
        batch = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in self.extensions:
                        continue
                    try:
                        # Skip 0-byte files
                        if not entry.is_file() or entry.stat().st_size == 0:
                            continue
                    except (FileNotFoundError, PermissionError, OSError):
                        continue

                    batch.append(os.path.normpath(entry.path))
                    if len(batch) >= self.chunk_size:
                        self._flush(directory, batch)
                        batch = []

            if batch:
                self._flush(directory, batch)

        except (PermissionError, OSError):
            print(f"Permission denied or error accessing directory: {directory}")


def make_queue():
    return queue.Queue()
