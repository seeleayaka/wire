"""Record the actual user reply '可以' to the preceding reference-image question."""
from datetime import datetime,timezone
import json
from pathlib import Path
from core import image_binding,sha256
from reference_confirmation import confirm_scope
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    draft_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    draft=json.loads(draft_path.read_text(encoding='utf-8'))
    binding=image_binding(draft['reference_image_path'])
    approval={'source':'human_user_message_in_current_chat','approved':True,'quote':'可以',
        'question':'左圖這束 CPU 風扇線接到 FAN_CPU，是否作為演示的正常參考？確認一次即可，不需要逐張標記。',
        'reviewer':'用户（当前聊天的参考确认）','client_date_hk':'2026-10-06',
        'reference_binding':binding,'expected_visible_attachment':['FAN_CPU','FAN_LEAD']}
    confirmed=confirm_scope(draft,binding,approval)
    output=ROOT/'artifacts/mendeley_reference_confirmed_20261006';output.mkdir(exist_ok=False)
    save(output/'reference_scope_confirmed.json',confirmed)
    save(output/'approval_record.json',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'approval':approval,'draft_sha256':sha256(draft_path),'draft_path':str(draft_path),
        'confirmed_scope_sha256':sha256(output/'reference_scope_confirmed.json'),
        'old_draft_retained_unmodified':True,'source_training_labels_not_human_confirmed':True,
        'electrical_continuity_not_confirmed':True,'mainline_pins':source_pins(),
        'code_pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('reference_confirmation.py')]}})
    print(json.dumps({'reference_review_confirmed':True,'electrical_continuity_confirmed':False,
        'confirmed_scope':str(output/'reference_scope_confirmed.json')},ensure_ascii=False))


if __name__=='__main__':main()
