"""Public official ONNX download. TLS verification and publisher SHA required."""
from pathlib import Path
import hashlib
import json
import sys
import requests
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/route_english_ocr_dependencies_20261005/python_tls'
REGISTRY='https://raw.githubusercontent.com/RapidAI/RapidOCR/main/python/rapidocr/default_models.yaml'
URL='https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx'
DIGEST='c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8'


def main():
    if OUT.exists():raise FileExistsError('preserve previous download attempt')
    OUT.mkdir(parents=True)
    try:
        # No API keys/cookies, private images, TLS disable, or paid service.
        session=requests.Session()
        response=session.get(REGISTRY,timeout=(15,60));response.raise_for_status();registry=response.content
        model=yaml.safe_load(registry)['onnxruntime']['PP-OCRv5']['rec']['en_PP-OCRv5_rec_mobile']
        if model['model_dir']!=URL or model['SHA256']!=DIGEST:raise ValueError('publisher model provenance changed')
        response=session.get(URL,timeout=(15,90));response.raise_for_status();binary=response.content
        actual=hashlib.sha256(binary).hexdigest()
        if actual!=DIGEST or not 1_000_000<len(binary)<64_000_000:raise ValueError('model hash/size mismatch')
        (OUT/'official_default_models.yaml').write_bytes(registry)
        (OUT/'en_PP-OCRv5_rec_mobile.onnx').write_bytes(binary)
        report=dict(status='downloaded_hash_verified',model_sha256=actual,model_bytes=len(binary),url=URL,
            registry_url=REGISTRY,registry_sha256=hashlib.sha256(registry).hexdigest(),TLS_verified=True,
            no_private_uploads=True,no_package_install_or_mainline_changes=True,model_compatibility_not_yet_tested=True)
        (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
    except BaseException as exc:
        (OUT/'failure.json').write_text(json.dumps(dict(status='failed',error_type=type(exc).__name__,no_TLS_bypass=True)),encoding='utf-8')
        raise


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
