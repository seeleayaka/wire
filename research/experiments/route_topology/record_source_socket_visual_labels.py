"""Record actual assistant qualitative source review, NEVER a human confirmation.

IDs below refer to the actual source-only contact sheets inspected in this chat.
This does not derive port state from dataset class strings or inspection names.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

from core import sha256
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    normal=ROOT/'artifacts/mendeley_normal_source_socket_review_20261005'
    extra=ROOT/'artifacts/mendeley_source_socket_occupancy_review_20261005'
    n=json.loads((normal/'review_inventory.json').read_text(encoding='utf-8'))
    e=json.loads((extra/'review_inventory.json').read_text(encoding='utf-8'))
    # Explicit reviewed source IDs; NOT an image-path classifier.
    visible={f'normal_source_{i:03d}' for i in range(1,121)}|{
        f'source_routing_{i:03d}' for i in [*range(1,32),*range(35,41)]}
    exposed={f'source_disconnection_{i:03d}' for i in range(1,41)}
    ambiguous={'source_routing_032','source_routing_033','source_routing_034'}
    inventory=n['rows']+e['rows']
    if {r['id'] for r in inventory}!=visible|exposed|ambiguous:raise ValueError('review ID inventory mismatch')
    pages=[*[normal/f'normal_source_page_{i}.png' for i in range(1,7)],
        *[extra/f'{pool}_page_{i}.png' for pool in ['source_routing','source_disconnection'] for i in [1,2]]]
    output=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005'
    if output.exists():raise FileExistsError('preserve prior annotations')
    rows=[]
    for row in inventory:
        if sha256(row['source_path'])!=row['source_sha256'] or sha256(row['patch_path'])!=row['patch_sha256']:
            raise ValueError('reviewed pixels changed')
        identity=row['id'];label=1 if identity in visible else 0 if identity in exposed else None
        note=('Visible dark mating-body profile with bundle entry above the contact row; no seating/continuity claim'
            if label==1 else 'Exposed socket-contact profile; no visible matching mating body, possible front occlusion'
            if label==0 else 'Paper occlusion hides enough of mating body to avoid a definite training label')
        rows.append({**row,'visual_label':label,'phenotype':'mating_body_visible' if label==1 else
            'socket_contacts_exposed' if label==0 else 'uncertain',
            'reviewer_type':'assistant_qualitative_visual','reviewer':'Codex source-photo visual review, NOT human certified',
            'evidence_note':note,'human_reviewed':False,'physical_seating_or_continuity_verified':False})
    output.mkdir(exist_ok=False)
    save(output/'labels.json',{'created_at':datetime.now(timezone.utc).isoformat(),'rows':rows,
        'counts':{'mating_body_visible':157,'socket_contacts_exposed':40,'uncertain':3},
        'label_status':'assistant_qualitative_source_annotations_not_certified_GT',
        'original_GT_files_modified':False,'inspection_labels_read_or_created':False,
        'reference_human_review_confirmed':False,'reviewed_contact_sheet_pins':{str(p):sha256(p) for p in pages},
        'source_pins':{str(p):sha256(p) for p in [Path(__file__),normal/'review_inventory.json',extra/'review_inventory.json']}})
    print(json.dumps({'source_annotation_status':'qualitative_not_certified_GT','visible':157,'exposed':40,'uncertain':3}))


if __name__=='__main__':main()
