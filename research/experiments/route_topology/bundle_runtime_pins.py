"""Same E-mainline fingerprints as run_audit, using only the standard library."""
import hashlib
from pathlib import Path
import subprocess


def source_pins():
    project = Path('E:/PythonProject10')
    files = ['inspection_agent/topology.py', 'inspection_agent/terminal_mapping.py',
             'inspection_agent/visible_segment_geometry.py', 'inspection_agent/sam_topology_adapter.py',
             'prototype/sam3_wire_fusion.py', 'inspection_agent/workflow.py',
             'prototype/assembly_auto_review_dino.py']
    pins = {str(project/path): hashlib.sha256((project/path).read_bytes()).hexdigest() for path in files}
    status = subprocess.check_output(['E:/Git/cmd/git.exe', '-C', str(project), 'status', '--porcelain'], text=True)
    pins['dirty_git_status_sha256'] = hashlib.sha256(status.encode()).hexdigest()
    return pins
