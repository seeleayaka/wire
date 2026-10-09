"""Bind accepted295 by actual output identity, not a legacy cache name."""
from replacement_voters import check_prefix


def check_accepted(legacy, accepted, research, trial):
    check_prefix(legacy, accepted, accepted)
    check_prefix(accepted, research, trial)
    return True


def check_totals(totals, legacy=False):
    original_tp=277 if legacy else 295
    for key,tp in [('original',original_tp),('accepted',295),('current',298)]:
        if key not in totals:
            if key=='accepted' and not legacy:continue
            raise ValueError('missing explicitly identified protected baseline')
        if totals[key] != dict(tp=tp,unmatched=4,fn=344-tp,predictions=tp+4,targets=344):
            raise ValueError('protected baseline identity/metrics mismatch: '+key)
    return True
