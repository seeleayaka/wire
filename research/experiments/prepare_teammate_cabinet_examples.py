"""Fixed original cabinet inputs for fresh teammate screenshots, no cherry-picking."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'artifacts/teammate_cabinet_examples_20261008_suite.json'


def main():
    if TARGET.exists():raise FileExistsError('preserve frozen selection')
    source=ROOT/'artifacts/local_performance_20261007_suite.json'
    suite=json.loads(source.read_text(encoding='utf-8'))
    cases=[next(c for c in suite['cases'] if c['id']==case) for case in ['cabinet_1','cabinet_2','cabinet_4_wrong3']]
    pins={}
    for c in cases:
        assert not c['ports'],'not Mendeley port branch'
        for field in ['reference','inspection']:
            path=Path(c[field]);assert path.is_file();pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    TARGET.write_text(json.dumps(dict(cases=cases,input_sha256=pins,
        selection='cabinet1/cabinet2 first frozen generated variants plus existing cabinet4 wrong3; fixed before new inference',
        old_intermediate_results_used=False,accuracy_boundary='reused generated development examples, not physical electrical truth'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(TARGET))


if __name__=='__main__':main()
