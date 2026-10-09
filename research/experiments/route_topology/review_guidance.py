"""Deterministic evidence-acquisition guidance; no inference or verdict override."""
from copy import deepcopy


def build_plan(scope, case, review=None):
    binding = case['image_binding']
    if not isinstance(binding, dict) or not binding.get('image_sha256'):
        raise ValueError('image-bound evidence required')
    automatic = deepcopy(case['automatic_comparison'])
    expected = {a['kind']: a['id'] for a in scope['anchors']}
    if set(expected) != {'visible_lead_emergence', 'wire_entry_socket'}:
        raise ValueError('two expected visible endpoints required')
    if review is not None and (review['case_id'] != case['id'] or review['image_binding'] != binding
                               or review['automatic_comparison_unchanged'] != automatic):
        raise ValueError('review must belong to the same immutable evidence')
    actions = []

    def add(code, instruction, reason):
        actions.append({'code': code, 'instruction': instruction, 'reason': reason,
                        'execution': 'not_executed', 'requires_operator': True})

    if review is None:
        add('review_endpoints', '在原图核对两端身份，并分别框出出线处与插口。',
            '自动结论不能代替本次操作者核查，框选本身也不证明身份正确。')
        add('review_bundle', '确认两端属于同一线束；若有遮挡，请保留未知。',
            '路径形状或距离接近不足以证明端点对应。')
        add('review_socket', '核对插接、接点露出或看不清，记录实际可见状态。',
            '插口外观证据与电气通断是不同检查。')
    else:
        s = review['human_review']
        decision = review['human_recheck_decision']
        if decision == 'insufficient_evidence':
            if not all(e['identity_confirmed'] for e in s['endpoints']):
                add('review_endpoints', '重新核对尚未确认的端点身份，不按距离猜测。',
                    '至少一端身份未确认。')
            if not s['same_bundle_confirmed']:
                add('review_bundle', '核查是否同一线束；遮挡处需另一视角或现场查看。',
                    '两端的同线束对应关系尚未确认。')
            if s['socket_state'] == 'uncertain':
                add('review_socket', '补拍插口近照或现场查看插接状态；无法取得则保持未知。',
                    '插口状态尚不清楚。')
            add('retain_unknown', '当前报告保持证据不足；不能用建议或重复图像补成确定结论。',
                '建议不是新观察，重复同一图像也不是独立证据。')
        else:
            add('review_record', '复核已记录的身份、框选与证据说明；必要时另存纠正记录。',
                '记录来源为' + review['review_source_label'] + '，身份未独立认证。')
            if decision in ('human_confirmed_changed_visible_attachment',
                            'human_confirmed_reference_socket_contacts_exposed'):
                add('inspect_on_site', '由操作者现场核查此可见差异；系统不自动执行修复。',
                    '可见变化不等同于已证实电气故障。')
    return {
        'schema_version': 1, 'kind': 'rule_based_review_guidance',
        'label': '复核建议（规则生成，非大模型识别）',
        'case_id': case['id'], 'image_binding': deepcopy(binding),
        'automatic_comparison_unchanged': automatic,
        'reference_basis': {'reference_binding': deepcopy(scope['reference_binding']),
                            'expected_endpoint_identities': expected,
                            'reference_confirmation': deepcopy(scope['reference_review']),
                            'basis_type': 'confirmed_visible_reference_not_wiring_diagram'},
        'review_source': None if review is None else review['evidence_source'],
        'actions': actions, 'automatic_new_hits': 0,
        'model_calls': 0, 'network_calls': 0, 'tools_executed': [],
        'electrical_continuity': 'not_assessed', 'deployed_to_E_mainline': False,
        'limitations': ['只核查可见线束关系，不是逐芯或电气导通检查。',
                        '未接入外部说明系统源码、知识库或大模型服务。',
                        '走线路径不同本身不是接法错误。'],
    }


def markdown(plan):
    return '# ' + plan['label'] + '\n\n病例：' + plan['case_id'] + '\n\n' + '\n\n'.join(
        a['instruction'] + '\n依据：' + a['reason'] + '\n状态：未执行，需要操作者。'
        for a in plan['actions']) + '\n\n' + '\n'.join(plan['limitations']) + '\n'
