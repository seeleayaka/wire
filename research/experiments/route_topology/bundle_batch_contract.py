"""Completeness/provenance guard for frozen same-device batch, not accuracy GT."""
def validate_preparation(rows, sam_cases, inspection_count=30):
    expected = ['reference'] + [f'case_{i+1:02d}' for i in range(inspection_count)]
    if [row['id'] for row in rows] != expected:
        raise ValueError('every inspection original must be included once in frozen order')
    if rows[0]['phenotype'] != 'mating_body_visible' or not rows[0]['sam_inference_requested']:
        raise ValueError('reference visible-body evidence and fresh SAM required')
    requested = []
    for row in rows:
        if row['sam_inference_requested']:
            if row['phenotype'] not in {'mating_body_visible', 'socket_contacts_exposed'}:
                raise ValueError('uncertain socket cannot enter definite bundle inference')
            if not row.get('crop_context') or 'descriptor' not in row or 'socket_patch_path' not in row:
                raise ValueError('exact original crop and qualified fresh socket features required')
            requested.append(row['id'])
    if [case['id'] for case in sam_cases] != requested:
        raise ValueError('SAM roster must match all and only eligible fresh prepared originals')
    return True
