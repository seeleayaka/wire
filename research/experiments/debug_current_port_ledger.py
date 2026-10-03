"""Read-only reproduction of historical versus formal recheck metadata."""
from current_port_baseline_audit import load, BASE, read_current_case
from inspection_agent.teacher_student_port_support import native_selection
import json
entry = load(BASE / 'train/report.json')['cases'][0]
teacher, record = read_current_case('train', entry, {})
old = record['current']
saved = old['primary'] + old['supplementary'] + old['zoom']
formal = native_selection(teacher)['all_predictions']
differences = []
for i, (a, b) in enumerate(zip(saved, formal)):
    if a != b:
        differences.append(dict(index=i, saved_only=sorted(set(a)-set(b)), formal_only=sorted(set(b)-set(a)),
                                unequal_common=[k for k in set(a)&set(b) if a[k]!=b[k]]))
print(json.dumps(dict(image=entry['image'], saved_length=len(saved), formal_length=len(formal), differences=differences), indent=2))
