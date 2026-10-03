"""Accept explicit safety abstention, never count it as model recognition."""


def validate_upstream(report, original, expected, *, normal_control=False):
    if original['status'] == 'applied':
        return 'applied'
    if (not normal_control or expected or original['status'] != 'fallback'
            or original.get('fallback_reason') != 'local_alignment_not_supported'):
        raise AssertionError('Unexpected upstream status: ' + str(original.get('fallback_reason')))
    if not any(item.get('dino_alignment_input') == 'local_ecc_corrected'
               for item in report.get('local_alignment', []) if isinstance(item, dict)):
        raise AssertionError('Local-alignment abstention must have actual correction provenance')
    if original.get('rescue_hints') or original.get('supplementary_hints'):
        raise AssertionError('Abstained control must not produce port hints')
    return 'safety_abstention_local_alignment'
