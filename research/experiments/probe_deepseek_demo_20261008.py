"""User-authorized official DeepSeek check; never persists or prints credentials."""
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'E:/PythonProject10/prototype')
import deepseek_mask_review as review


def main():
    out = ROOT / 'artifacts/deepseek_demo_probe_20261008'
    out.mkdir(exist_ok=False)
    source = Path('C:/Users/HUAWEI/Desktop/中转站信息.txt').read_text(encoding='utf-8')
    urls = list(re.finditer(r'https://api\.deepseek\.com[^\s"\']*', source))
    if len(urls) != 1:
        raise ValueError('Expected exactly one official DeepSeek URL; no credential guessing')
    nearby = source[max(0, urls[0].start() - 300):urls[0].start()]
    keys = re.findall(r'sk-[A-Za-z0-9_-]+', nearby)
    if len(keys) != 1:
        raise ValueError('Ambiguous endpoint-key pairing; no credential guessing')
    key = keys[0]
    origin = 'https://' + urlsplit(urls[0].group()).netloc
    assert origin == 'https://api.deepseek.com'
    settings = review.load_settings()
    result = {'credential_saved': False, 'raw_images_sent': False,
              'configured_model': settings.model, 'production_modified': False}
    def save():
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        assert key not in serialized
        (out / 'report.json').write_text(serialized, encoding='utf-8')
    try:
        request = Request(origin + '/models', headers={'Authorization': 'Bearer ' + key})
        with urlopen(request, timeout=20) as response:
            models = json.loads(response.read())
        result['models'] = [m['id'] for m in models.get('data', [])]
        result['model_list_status'] = 'ok'
    except HTTPError as error:
        result['model_list_status'] = 'http_' + str(error.code)
        save()
        print(json.dumps(result, ensure_ascii=False))
        return
    except (URLError, TimeoutError, OSError):
        result['model_list_status'] = 'network_unavailable_or_timeout'
        save()
        print(json.dumps(result, ensure_ascii=False))
        return
    save()
    print(json.dumps({'model_list_status': 'ok', 'models': result['models']}, ensure_ascii=False), flush=True)
    path = ROOT / 'artifacts/local_performance_fresh_20261007/cabinet_2/desktop_output/20261007_200359/report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    fusion = report['sam3_fusion']
    ref = Path(fusion['reference_sam3']['output_dir']) / 'mask_union.png'
    ins = Path(fusion['inspection_sam3']['output_dir']) / 'mask_union.png'
    settings = replace(settings, endpoint=origin + '/chat/completions', timeout_seconds=45)
    # Actual existing GUI backend, existing configured model, one mask-only
    # question. No endpoint-key overwrite and no switch to a text model disguised
    # as successful vision inference.
    try:
        result['mask_review'] = review.ask_masks(
            ref, ins, report['review_regions'],
            '请说明两个候选区应如何人工复核；仅凭掩膜不能确认哪些电气结论？',
            settings=settings, api_key=key,
        )
        result['mask_review_status'] = 'ok'
    except review.DeepSeekMaskReviewError as error:
        result['mask_review_status'] = 'failed'
        result['safe_error'] = str(error).replace(key, '[REDACTED]')
    save()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
