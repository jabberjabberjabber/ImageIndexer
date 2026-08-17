"""Everything that touches files on disk: ExifTool reads/writes, sidecar
resolution, and repair operations (renaming invalid files, fixing
extensions).

All ExifTool calls go through the single MetadataStore instance and are made
from the pipeline's main thread only -- ExifToolHelper is a stateful
subprocess and is not thread-safe.
"""
import os

import exiftool

KEYWORD_FIELDS = frozenset((
    "Keywords", "IPTC:Keywords", "Composite:keywords",
    "Subject", "DC:Subject", "XMP:Subject", "XMP-dc:Subject",
))
CAPTION_FIELDS = frozenset((
    "Description", "XMP:Description", "ImageDescription", "DC:Description",
    "EXIF:ImageDescription", "Composite:Description", "Caption",
    "IPTC:Caption", "Composite:Caption", "IPTC:Caption-Abstract",
    "XMP-dc:Description", "PNG:Description",
))
IDENTIFIER_FIELDS = frozenset(("Identifier", "XMP:Identifier"))
STATUS_FIELDS = frozenset(("Status", "XMP:Status"))
ORIENTATION_FIELD = "EXIF:Orientation"

READ_FIELDS = (list(KEYWORD_FIELDS) + list(CAPTION_FIELDS)
               + list(IDENTIFIER_FIELDS) + list(STATUS_FIELDS)
               + ["File:FileType", "File:FileTypeExtension", ORIENTATION_FIELD])


class MetadataStore:

    def __init__(self, config, callback=print):
        self.config = config
        self.callback = callback
        print("Initializing ExifTool...")
        self.et = exiftool.ExifToolHelper(encoding="utf-8")
        print("ExifTool initialized successfully")
        # Maps rebuilt per batch by get_batch()
        self._read_to_image = {}
        self._write_target = {}

    def sidecar_path_for_image(self, image_path):
        if self.config.no_sidecar_extension:
            return os.path.splitext(image_path)[0] + ".xmp"
        return image_path + ".xmp"

    def find_existing_sidecar(self, image_path):
        candidates = [image_path + ".xmp",
                      os.path.splitext(image_path)[0] + ".xmp"]
        if self.config.no_sidecar_extension:
            candidates.reverse()
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def resolve_metadata_source(self, image_path):
        """Determine where this image's metadata is read from and written to:

           - An existing sidecar is authoritative for BOTH read and write,
             regardless of use_sidecar. We never write the image behind a
             sidecar's back.
           - No sidecar + use_sidecar on: read the image (migrates embedded
             tags for free), write a new sidecar.
           - No sidecar + use_sidecar off: image is the only store.
           - Validation always targets the image; a sidecar can't report
             that the image is corrupt.

           Returns (read_target, write_target, validate_target).
        """
        existing = self.find_existing_sidecar(image_path)
        if existing:
            return existing, existing, image_path
        if self.config.use_sidecar:
            return image_path, self.sidecar_path_for_image(image_path), image_path
        return image_path, image_path, image_path

    def image_for_read_target(self, read_target):
        return self._read_to_image.get(os.path.normpath(read_target))

    def write_target_for(self, image_path):
        return self._write_target.get(os.path.normpath(image_path), image_path)

    def get_batch(self, files):
        """Read metadata for a batch of images (or their sidecars).

        Sidecar-backed images also get the image itself validated, since the
        sidecar can't report image corruption.
        """
        self._read_to_image = {}
        self._write_target = {}
        read_targets, need_val = [], []

        for img in files:
            read_t, write_t, val_t = self.resolve_metadata_source(img)
            read_targets.append(read_t)
            self._write_target[os.path.normpath(img)] = write_t
            if os.path.normpath(read_t) != os.path.normpath(img):
                self._read_to_image[os.path.normpath(read_t)] = img
            if (not self.config.skip_verify
                    and os.path.normpath(val_t) != os.path.normpath(read_t)):
                need_val.append((read_t, val_t))

        try:
            params = [] if self.config.skip_verify else ["-validate"]
            metadata_list = self.et.get_tags(read_targets, tags=READ_FIELDS,
                                             params=params)

            if need_val:
                images = [img for _r, img in need_val]
                vals = self.et.get_tags(images, tags=["ExifTool:Validate"],
                                        params=["-validate"])
                val_by_img = {
                    os.path.normpath(v["SourceFile"]): v.get("ExifTool:Validate", "0 0 0")
                    for v in vals
                }
                read_to_img = {os.path.normpath(r): img for r, img in need_val}
                for m in metadata_list:
                    img = read_to_img.get(os.path.normpath(m["SourceFile"]))
                    if img is not None:
                        m["ExifTool:Validate"] = val_by_img.get(
                            os.path.normpath(img), "0 0 0")

            return metadata_list

        except exiftool.exceptions.ExifToolExecuteError as e:
            print(f"ExifTool Execute Error: {str(e)}")
            self.callback("ExifTool execute error - check if files are accessible")
        except exiftool.exceptions.ExifToolVersionError as e:
            print(f"ExifTool Version Error: {str(e)}")
            print("  Please update ExifTool to a compatible version")
        except Exception as e:
            print(f"ExifTool Error: {type(e).__name__} - {str(e)}")
        return []

    def write(self, file_path, metadata, on_write_error=None):
        """Write tags to the resolved target for file_path.

        on_write_error(image_path, write_target, writing_sidecar) is invoked
        on failure so the pipeline can decide about renaming/cleanup.
        """
        if self.config.dry_run:
            print("Dry run. Not writing.")
            return True

        write_target = self.write_target_for(file_path)
        writing_sidecar = write_target.lower().endswith(".xmp")

        try:
            params = ["-m"]
            if self.config.preserve_date:
                params.append("-P")
            if self.config.no_backup or writing_sidecar:
                params.append("-overwrite_original")

            self.et.set_tags(write_target, tags=metadata, params=params)
            return True

        except Exception as e:
            print(f"Metadata Write Error: {os.path.basename(file_path)}")
            print(f"  {type(e).__name__}: {str(e)}")
            self.callback(f"\nError writing metadata: {str(e)}")
            if on_write_error:
                on_write_error(file_path, write_target, writing_sidecar)
            return False

    def terminate(self):
        self.et.terminate()

def rename_to_invalid(file_path, callback=print):
    """Rename a file to filename_ext.invalid. Returns True on success."""
    try:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        # Clean up exiftool temp files: filename_exiftool_tmp
        exiftool_tmp = file_path + "_exiftool_tmp"
        if os.path.exists(exiftool_tmp):
            try:
                os.remove(exiftool_tmp)
                callback(f"Cleaned up temporary file: {os.path.basename(exiftool_tmp)}")
            except Exception as e:
                callback(f"Could not remove temp file "
                         f"{os.path.basename(exiftool_tmp)}: {str(e)}")

        # Backup files (_original) from runs without -overwrite_original
        
        backup_file = file_path + "_original"
        if os.path.exists(backup_file):
            if not os.path.exists(file_path):
                # Main file gone; the backup IS the file to rename
                file_path = backup_file
                base_name = os.path.basename(backup_file)
            else:
                try:
                    os.remove(backup_file)
                    callback(f"Cleaned up backup file: {os.path.basename(backup_file)}")
                except Exception as e:
                    callback(f"Could not remove backup file "
                             f"<{os.path.basename(backup_file)}>: {str(e)}")

        
        if not os.path.exists(file_path):
            callback(f"File no longer exists, cannot rename: {base_name}")
            return False

        # filename_ext.invalid, or filename_ext(N).invalid for duplicates
        name_parts = base_name.rsplit(".", 1)
        base_invalid_name = (f"{name_parts[0]}_{name_parts[1]}"
                             if len(name_parts) == 2 else base_name)

        new_path = os.path.join(dir_name, f"{base_invalid_name}.invalid")
        original_new_path = new_path
        counter = 1
        while os.path.exists(new_path):
            new_path = os.path.join(dir_name, f"{base_invalid_name}({counter}).invalid")
            counter += 1
            if counter > 1000:
                callback(f"Too many duplicate .invalid files, cannot rename: {base_name}")
                return False

        os.rename(file_path, new_path)
        suffix = " (duplicate name)" if new_path != original_new_path else ""
        callback(f"Renamed invalid file: {base_name} -> "
                 f"{os.path.basename(new_path)}{suffix}")
        print(f"Invalid or corrupt file <{base_name}> renamed to "
              f"<{os.path.basename(new_path)}>{suffix}")
        return True

    except Exception as e:
        callback(f"Failed to rename invalid file <{file_path}>: {str(e)}")
        print(f"Failed: {str(e)}")
        return False


def fix_file_extension(file_path, expected_ext, callback=print):
    """Rename file_path so its extension matches expected_ext.
    Returns the (possibly new) path.
    """
    if not expected_ext:
        return file_path

    expected_ext = expected_ext.lower()
    if not expected_ext.startswith("."):
        expected_ext = "." + expected_ext

    base_path, current_ext = os.path.splitext(file_path)
    if current_ext.lower() == expected_ext:
        return file_path

    try:
        new_path = base_path + expected_ext
        counter = 1
        while os.path.exists(new_path):
            new_path = f"{base_path}({counter}){expected_ext}"
            counter += 1
            if counter > 1000:
                callback(f"Too many files with same name, cannot rename: "
                         f"{os.path.basename(file_path)}")
                return file_path

        os.rename(file_path, new_path)
        msg = (f"Fixed extension: {os.path.basename(file_path)} -> "
               f"{os.path.basename(new_path)}")
        print(msg)
        callback(msg)
        return new_path

    except Exception as e:
        callback(f"Failed to fix extension for {file_path}: {str(e)}")
        print(f"Extension fix failed: {str(e)}")
        return file_path
