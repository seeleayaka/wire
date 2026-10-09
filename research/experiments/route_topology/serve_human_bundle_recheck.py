"""Loopback-only actual demo review; append-only submissions, no model launch."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import shutil
import uuid
from urllib.parse import urlsplit
from core import image_binding, sha256
from human_bundle_recheck import recheck
from visible_lead_scope import validate_scope

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(output):
    folder = ROOT / 'artifacts/mendeley_confirmed_bundle_demo_20261006'
    report_path = ROOT / 'artifacts/mendeley_confirmed_bundle_review_20261006/report.json'
    scope_path = ROOT / 'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    prep_path = folder / 'preparation_report.json'
    report, prep, scope = read(report_path), read(prep_path), read(scope_path)
    if report.get('status') != 'complete':
        raise ValueError('completed source report required')
    pins = dict(report['pins'])
    pins.update({str(p): sha256(p) for p in (report_path, prep_path, scope_path, Path(__file__), Path(__file__).with_name('human_bundle_recheck.py'), Path(__file__).with_name('human_bundle_recheck.html'))})
    for p, digest in pins.items():
        if sha256(p) != digest:
            raise ValueError('source evidence drift: ' + p)
    if image_binding(scope['reference_image_path']) != scope['reference_binding'] or not validate_scope(scope, scope['reference_binding']):
        raise ValueError('reference identity drift')
    rows = {r['id']: r for r in prep['cases']}
    items = []
    for c in report['cases']:
        original = rows[c['id']]['original_source']
        binding = image_binding(original['path'])
        if any(binding[k] != original[k] for k in binding):
            raise ValueError('inspection original drift')
        items.append({'id': c['id'], 'image_binding': binding, 'source_path': original['path'],
                      'automatic_comparison': c['comparison'], 'image_url': '/image/' + c['id']})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    shutil.copyfile(scope['reference_image_path'], output / 'images/reference.jpg')
    for c in items:
        shutil.copyfile(c['source_path'], output / ('images/' + c['id'] + '.jpg'))
    catalog = {'scope': scope, 'cases': items, 'source_pins': pins,
               'source_report_sha256': sha256(report_path), 'fresh_inference': False,
               'historical_evidence_explicitly_reused': True, 'deployed': False}
    (output / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    return catalog


def handler(catalog, output, token, port):
    allowed_host = '127.0.0.1:' + str(port)
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value, mime='application/json; charset=utf-8'):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host') != allowed_host:
                return self.respond(403, {'error': 'invalid host'})
            path = urlsplit(self.path).path
            if path == '/':
                return self.respond(200, Path(__file__).with_name('human_bundle_recheck.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/catalog':
                return self.respond(200, dict(catalog, csrf_token=token))
            images = {'/image/reference': output / 'images/reference.jpg'}
            images.update({c['image_url']: output / ('images/' + c['id'] + '.jpg') for c in catalog['cases']})
            if path in images:
                return self.respond(200, images[path].read_bytes(), 'image/jpeg')
            return self.respond(404, {'error': 'not found'})

        def do_POST(self):
            if (self.headers.get('Host') != allowed_host or self.headers.get('Origin') != 'http://' + allowed_host
                    or self.headers.get('X-Review-Token') != token):
                # Drain only a bounded body so Windows does not reset the client
                # before it can read the rejection. Never parse or save it.
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if 0 < size <= 16384:
                        self.rfile.read(size)
                except ValueError:
                    pass
                return self.respond(403, {'error': 'local review token/origin required'})
            if self.path != '/recheck':
                return self.respond(404, {'error': 'not found'})
            try:
                count = int(self.headers.get('Content-Length', '0'))
                if not 0 < count <= 16384:
                    raise ValueError('request too large or empty')
                submission = json.loads(self.rfile.read(count))
                if not isinstance(submission, dict):
                    raise ValueError('review must be an object')
                for p, digest in catalog['source_pins'].items():
                    if sha256(p) != digest:
                        raise ValueError('source evidence changed; reopen a new review session')
                case = next((c for c in catalog['cases'] if c['id'] == submission.get('case_id')), None)
                if case is None:
                    raise ValueError('unknown case')
                if image_binding(case['source_path']) != case['image_binding'] or image_binding(output / ('images/' + case['id'] + '.jpg')) != case['image_binding']:
                    raise ValueError('inspection photo changed')
                scope = catalog['scope']
                if image_binding(scope['reference_image_path']) != scope['reference_binding'] or image_binding(output / 'images/reference.jpg') != scope['reference_binding']:
                    raise ValueError('reference photo changed')
                result = recheck(scope, case, submission)
                result['source_report_sha256'] = catalog['source_report_sha256']
                dest = output / ('review_' + uuid.uuid4().hex)
                dest.mkdir(exist_ok=False)
                (dest / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                summary = '# 可见线束复核记录\n\n' + '\n\n'.join([
                    '病例：' + case['id'], '原自动结论：' + case['automatic_comparison']['decision'],
                    '记录来源：' + result['review_source_label'],
                    '复核结论：' + result['review_result_label'],
                    '复核人（自行填写，未独立认证）：' + result['reviewer'], '说明：' + result['evidence_note'],
                    '补证单独记录；自动新增命中 0；本次没有重新运行 SAM。',
                    '仅为可见多芯线束关系，不证明逐芯接线、电气导通、隐藏端子或实物修复。'])
                (dest / 'report.md').write_text(summary, encoding='utf-8')
                self.respond(200, {'report': result, 'saved_report': str(dest / 'report.json')})
            except (ValueError, TypeError, KeyError) as e:
                self.respond(400, {'error': str(e)})

        def log_message(self, *_):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8767)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('new output must stay under workspace artifacts')
    catalog = prepare(output)
    server = HTTPServer(('127.0.0.1', args.port), handler(catalog, output, secrets.token_hex(32), args.port))
    print('READY http://127.0.0.1:' + str(args.port), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
