"""Verify cached CLI geometry and deterministic one-image scoring replay."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path('E:/PythonProject10'); sys.path.insert(0, str(ROOT))
from inspection_agent.sam_topology_adapter import connection_graph_from_sam_endpoints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry-dir', type=Path, required=True)
    p.add_argument('--cache-audit', type=Path, required=True)
    p.add_argument('--pilot-audit', type=Path, required=True)
    p.add_argument('--pilot-replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError('fresh output required')
    load=lambda path: json.loads(path.read_text(encoding='utf-8'))
    audit=load(args.cache_audit); summaries=[]
    for name, total, eligible in (('normal_003',23,16),('damaged_001',16,13),
                                 ('disconnected_001',62,37),('disconnected_002',15,12)):
        report=load(args.geometry_dir/name/'endpoint_report.json')
        records=report['records']; assert len(records)==total
        assert sum(r['geometry_pair_eligible'] for r in records)==eligible
        for r in records:
            if not r['geometry_pair_eligible']: assert r['visible_ends_xy']==[]
            else:
                assert r['candidate_tip_count']==2 and r['branch_cluster_count']==0
                assert r['visible_ends_xy']==r['candidate_tips_xy']
        # No invented terminal ROIs; all real records remain unresolved here.
        graph, adapter=connection_graph_from_sam_endpoints(report, {
            'schema_version':1,'graph_id':'undeclared-terminal-audit',
            'scene_type':'computer_interior','terminals':[]})
        assert not graph.connections
        assert adapter['summary']['legacy_geometry_unverified_record_count']==0
        if name!='disconnected_002':
            expected=next(s for s in audit['samples'] if Path(s['image']).stem==name)
            assert expected['component_count']==total and expected['open_unbranched_components']==eligible
        summaries.append({'image':name,'components':total,'pair_eligible':eligible,
                          'abstentions':total-eligible,'connections_without_declared_terminals':0})
    first, second=load(args.pilot_audit),load(args.pilot_replay)
    assert first==second, 'cached scoring/registration replay drift'
    result={'geometry_reports':summaries,'training_components':101,'training_pair_eligible':66,
            'training_abstentions':35,'pilot_report_exact_replay':True,
            'no_inference_in_replay':True,'not_claimed':'Cable identity, electrical continuity, field accuracy, dataset-wide precision gain.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))


if __name__=='__main__':main()
