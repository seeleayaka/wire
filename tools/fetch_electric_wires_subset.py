"""Fetch a small paired Electric Wires subset from a range-readable ZIP.

This deliberately never downloads the full archive.  The ZIP central-directory
catalog must have been produced first by inspect_electric_wires_zip_catalog.py.
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def extract_member(blob: bytes, compressed_size: int) -> bytes:
    if blob[:4] != b"PK\x03\x04":
        raise ValueError("range response does not start at a ZIP local header")
    _, _, _, compression, _, _, _, _, _, name_size, extra_size = struct.unpack_from(
        "<4sHHHHHIIIHH", blob, 0
    )
    start = 30 + name_size + extra_size
    payload = blob[start : start + compressed_size]
    if len(payload) != compressed_size:
        raise ValueError("range response ended before the compressed member")
    if compression == 0:
        return payload
    if compression == 8:
        return zlib.decompress(payload, -zlib.MAX_WBITS)
    raise ValueError(f"unsupported ZIP compression: {compression}")


def fetch(entry: dict, archive_url: str) -> bytes:
    start = int(entry["local_header_offset"])
    # Local header names/extra fields are short.  The extractor consumes only
    # the member bytes declared in the central directory.
    end = start + int(entry["compressed_size"]) + 1023
    request = urllib.request.Request(archive_url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if getattr(response, "status", None) != 206:
            raise RuntimeError(f"archive did not honour range request for {entry['name']}: {response.status}")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-"):
            raise RuntimeError(f"unexpected Content-Range for {entry['name']}: {content_range}")
        return extract_member(response.read(), int(entry["compressed_size"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent paired range requests; keep this small.")
    parser.add_argument(
        "--archive-url",
        default="https://amsacta.unibo.it/id/eprint/6654/1/REMODEL.zip",
    )
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in catalog["entries"]}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    def fetch_pair(sample_id: int) -> list[dict]:
        pair_written: list[dict] = []
        for kind, suffix in (("imgs", "rgb"), ("masks", "mask")):
            member_name = f"train/train/{kind}/{sample_id}.png"
            entry = entries.get(member_name)
            if entry is None:
                raise KeyError(f"missing archive member: {member_name}")
            destination = output / f"{sample_id}_{suffix}.png"
            payload = fetch(entry, args.archive_url)
            destination.write_bytes(payload)
            with Image.open(destination) as image:
                image.verify()
            pair_written.append({"id": sample_id, "kind": suffix, "archive_member": member_name, "bytes": len(payload)})
            print(f"{member_name} -> {destination.name} ({len(payload)} bytes)")
        return pair_written

    written: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 3))) as pool:
        for pair_written in pool.map(fetch_pair, args.ids):
            written.extend(pair_written)

    (output / "subset_manifest.json").write_text(
        json.dumps({"ids": args.ids, "members": written}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
