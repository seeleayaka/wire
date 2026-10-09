"""One bounded evidence-first replay after source completion, no model launches."""
import os
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha


def main():
    import psutil
    out = ROOT / 'artifacts/sparse_nomination_audit_queue_20261004'
    source = ROOT / 'artifacts/sparse_nomination_20261004'
    auditor = ROOT / 'experiments/audit_sparse_nomination.py'
    if out.exists():
        raise FileExistsError('Preserve single evidence-first audit queue')
    out.mkdir()
    digest = sha(auditor)
    start = time.monotonic()
    while not (source / 'report.json').exists():
        assert sha(auditor) == digest
        progress = load(source / 'progress.json')
        if progress['status'] == 'failed':
            save(out / 'report.json', dict(status='source_failed_no_audit', source_progress=progress))
            return
        if time.monotonic()-start > 7200:
            raise TimeoutError('Bounded audit wait expired; no inference launched')
        if not psutil.pid_exists(progress['pid']):
            raise RuntimeError('Source worker exited without final report')
        save(out / 'progress.json', dict(status='waiting_for_source', pid=os.getpid(), source_progress=progress))
        time.sleep(10)
    assert sha(auditor) == digest
    save(out / 'progress.json', dict(status='auditing', pid=os.getpid(), auditor_sha256=digest))
    code = subprocess.call([str(REPO / '.venv/Scripts/python.exe'), '-B', str(auditor)], cwd=str(ROOT))
    result = dict(status='audit_complete' if code == 0 else 'audit_failed', exit_code=code,
                  no_training=True, no_detector_or_DINO_inference=True, no_deployment=True,
                  source_report_sha256=sha(source / 'report.json'), auditor_sha256=digest)
    save(out / 'report.json', result)
    save(out / 'progress.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
