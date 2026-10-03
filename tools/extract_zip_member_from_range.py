"""Extract one deflated ZIP member from a byte-range response.

Use with the manifest produced by inspect_electric_wires_zip_catalog.py.  The
range must begin at the member's local header and include its compressed bytes.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-file", type=Path, required=True)
    parser.add_argument("--compressed-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blob = args.range_file.read_bytes()
    if blob[:4] != b"PK\x03\x04":
        raise ValueError("range does not start with a ZIP local header")
    _, _, _, compression, _, _, _, _, _, filename_size, extra_size = struct.unpack_from(
        "<4sHHHHHIIIHH", blob, 0
    )
    payload_start = 30 + filename_size + extra_size
    payload = blob[payload_start : payload_start + args.compressed_size]
    if len(payload) != args.compressed_size:
        raise ValueError("range does not include the complete compressed member")
    if compression == 0:
        image = payload
    elif compression == 8:
        image = zlib.decompress(payload, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported ZIP compression method: {compression}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    print(f"{args.output}: {len(image)} bytes")


if __name__ == "__main__":
    main()
