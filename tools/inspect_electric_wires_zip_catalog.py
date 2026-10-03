"""Inspect the central directory of a range-fetched ZIP without downloading it all.

The Electric Wires archive is 36.6 GiB, but its ZIP central directory is near
the archive end.  This utility reads a small tail range, reports the archive
layout and emits a manifest with precise byte spans for targeted sample files.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path


SIG_ZIP64_EOCD = b"PK\x06\x06"
SIG_CENTRAL = b"PK\x01\x02"


def _zip64_values(extra: bytes, wanted: list[bool]) -> list[int | None]:
    """Return replacement values from the ZIP64 extra field in ZIP spec order."""
    position = 0
    payload: bytes | None = None
    while position + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, position)
        position += 4
        field = extra[position : position + size]
        position += size
        if field_id == 0x0001:
            payload = field
            break

    if payload is None:
        return [None for _ in wanted]

    position = 0
    values: list[int | None] = []
    for use_zip64 in wanted:
        if use_zip64:
            if position + 8 > len(payload):
                raise ValueError("truncated ZIP64 extra field")
            values.append(struct.unpack_from("<Q", payload, position)[0])
            position += 8
        else:
            values.append(None)
    return values


def parse_central_directory(tail: bytes, tail_start: int) -> tuple[dict, list[dict]]:
    zip64_at = tail.rfind(SIG_ZIP64_EOCD)
    if zip64_at < 0:
        raise ValueError("ZIP64 end-of-central-directory record not found in range")

    # Signature, record size, versions, disk fields, entries-on-disk,
    # entries-total, central-directory size, central-directory offset.
    fields = struct.unpack_from("<4sQHHIIQQQQ", tail, zip64_at)
    _, _, _, _, _, _, _, entries_total, cd_size, cd_offset = fields
    relative = cd_offset - tail_start
    if relative < 0 or relative + cd_size > len(tail):
        raise ValueError(
            "range does not include the full central directory: "
            f"offset={cd_offset}, size={cd_size}, tail=[{tail_start}, {tail_start + len(tail)})"
        )

    entries: list[dict] = []
    position = relative
    for index in range(entries_total):
        if tail[position : position + 4] != SIG_CENTRAL:
            raise ValueError(f"central-directory entry {index} has an invalid signature")
        values = struct.unpack_from("<4sHHHHHHIIIHHHHHII", tail, position)
        (
            _,
            _,
            _,
            flag,
            compression,
            _,
            _,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_size,
            extra_size,
            comment_size,
            _,
            _,
            _,
            local_offset,
        ) = values
        name_start = position + 46
        name_bytes = tail[name_start : name_start + filename_size]
        extra_start = name_start + filename_size
        extra = tail[extra_start : extra_start + extra_size]
        replacements = _zip64_values(
            extra,
            [
                uncompressed_size == 0xFFFFFFFF,
                compressed_size == 0xFFFFFFFF,
                local_offset == 0xFFFFFFFF,
            ],
        )
        if replacements[0] is not None:
            uncompressed_size = replacements[0]
        if replacements[1] is not None:
            compressed_size = replacements[1]
        if replacements[2] is not None:
            local_offset = replacements[2]
        name = name_bytes.decode("utf-8" if flag & 0x800 else "cp437")
        entries.append(
            {
                "name": name,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_offset,
                "compression": compression,
                "crc32": f"{crc32:08x}",
            }
        )
        position = extra_start + extra_size + comment_size

    return {
        "entries": entries_total,
        "central_directory_offset": cd_offset,
        "central_directory_size": cd_size,
    }, entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--tail-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info, entries = parse_central_directory(args.tail.read_bytes(), args.tail_start)
    top_levels = Counter(item["name"].split("/", 1)[0] for item in entries)
    extensions = Counter(Path(item["name"]).suffix.lower() or "<none>" for item in entries)
    payload = {
        "archive": info,
        "top_level_counts": dict(top_levels.most_common()),
        "extension_counts": dict(extensions.most_common()),
        "first_100_names": [item["name"] for item in entries[:100]],
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: payload[k] for k in ("archive", "top_level_counts", "extension_counts")}, ensure_ascii=False, indent=2))
    for item in entries[:30]:
        print(f"{item['name']}\t{item['compressed_size']} bytes\toffset {item['local_header_offset']}")


if __name__ == "__main__":
    main()
