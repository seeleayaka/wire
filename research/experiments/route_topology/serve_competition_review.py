"""Isolated companion with complete reference-bound downloadable reports."""
import argparse
import json
from pathlib import Path
import secrets
from http.server import HTTPServer
from core import sha256
import serve_human_bundle_recheck as legacy_server
from human_bundle_recheck_bound import recheck
from serve_review_guidance import companion_handler

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8771)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(legacy_server.ROOT/'artifacts'):
        raise ValueError('output must stay under artifacts')
    # Override only this newly started process's dependency. Existing servers,
    # their frozen files, saved reports, and mainline remain untouched.
    legacy_server.recheck = recheck
    catalog = legacy_server.prepare(output)
    for name in ['serve_competition_review.py', 'human_bundle_recheck_bound.py',
                 'serve_review_guidance.py', 'review_guidance.py']:
        p = Path(__file__).with_name(name)
        catalog['source_pins'][str(p)] = sha256(p)
    catalog['standalone_reference_binding'] = True
    (output/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    server = HTTPServer(('127.0.0.1',args.port),companion_handler(catalog,output,secrets.token_hex(32),args.port))
    print('READY http://127.0.0.1:'+str(args.port),flush=True)
    server.serve_forever()

if __name__ == '__main__': main()
