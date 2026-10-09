"""Record an explicit human reference-scope confirmation without changing pixels."""
from copy import deepcopy
from core import text
from visible_lead_scope import validate_scope


def confirm_scope(draft,binding,approval):
    validate_scope(draft,binding)
    if approval.get('source')!='human_user_message_in_current_chat' or approval.get('approved') is not True:
        raise ValueError('explicit current human message approval required')
    if approval.get('reference_binding')!=binding or approval.get('expected_visible_attachment')!=draft['expected_visible_attachment']:
        raise ValueError('approval must bind this reference image and this attachment')
    for field in ['quote','question','reviewer','client_date_hk']:
        text(approval.get(field),field)
    result=deepcopy(draft)
    result['reference_review']={'confirmed':True,'reviewer':approval['reviewer'],
        'evidence_note':approval['question']+' User reply: '+approval['quote'],
        'confirmation_source':approval['source'],'client_date_hk':approval['client_date_hk'],
        'visible_reference_scope_only':True,'electrical_continuity_confirmed':False}
    result['annotation_origin']='assistant_reference_draft_explicitly_confirmed_by_current_human_message'
    validate_scope(result,binding)
    return result
