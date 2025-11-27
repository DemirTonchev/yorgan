from pathlib import Path
import hashlib


def generate_hashed_filename(filename: Path | str, content: bytes, digest_size=4) -> str:
    """Generates deterministic output. Collision probability depends on digest_size.
    Outputs mime type guessing friendly filename.
    """
    filename = Path(filename)
    hex_str = hashlib.blake2b(str(filename).encode() + content, digest_size=digest_size).hexdigest()
    filename_key = filename.stem + "-" + hex_str + filename.suffix
    return filename_key

