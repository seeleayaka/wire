"""One-shot bounded audit after the already-running worker; no model restart."""
import json
import subprocess
import time
from pathlib import Path
from run_prompt_contrast import save, digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/sam_prompt_confidence_source_20261008'
PYTHON = 'E:/PythonProject10/.venv/Scripts/python.exe'


def main():
    started = time.monotonic()
    own = digest(__file__)
    helpers = ['sam_prompt_confidence_trial.py', 'audit_sam_prompt_source_gate.py']
    pinned = {name: digest(ROOT / 'experiments/route_topology' / name) for name in helpers}
    try:
        while time.monotonic() - started < 5700:
            try:
                p = json.loads((OUT / 'progress.json').read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                time.sleep(2)
                continue
            if p['status'] == 'failed':
                raise RuntimeError('upstream inference failed; no restart: ' + p.get('error',''))
            if p['status'] == 'complete':
                break
            time.sleep(15)
        else:
            raise TimeoutError('bounded audit watcher timeout, worker not killed')
        assert digest(__file__) == own
        assert all(digest(ROOT / 'experiments/route_topology' / n) == sha for n,sha in pinned.items())
        for args in [['sam_prompt_confidence_trial.py','audit'], ['audit_sam_prompt_source_gate.py']]:
            command = [PYTHON, '-B', str(ROOT / 'experiments/route_topology' / args[0]), *args[1:]]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
            save(OUT / (args[0] + '.execution.json'), dict(returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr, fresh_model_calls=0))
            if result.returncode:
                raise RuntimeError('native audit failed: ' + args[0])
        gate = json.loads((OUT / 'source_gate_report.json').read_text(encoding='utf-8'))
        save(OUT / 'pipeline_report.json', dict(status='complete',
            any_strict_source_gate_passed=any(r['strict_source_gate_passed'] for r in gate['rows']),
            actual_visual_review='pending', no_deployment=True, no_demo_extension=True,
            gate_report_sha256=digest(OUT / 'source_gate_report.json'),
            electrical_connections_confirmed=0, not_field_accuracy=True))
        print('native audit and source gate complete; actual visual review remains required', flush=True)
    except BaseException as e:
        save(OUT / 'pipeline_report.json', dict(status='failed', error=str(e), no_restart=True))
        raise


if __name__ == '__main__': main()
