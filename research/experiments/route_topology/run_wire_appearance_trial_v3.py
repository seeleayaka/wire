from pathlib import Path
import json
import run_wire_appearance_trial as runner
from reference_wire_appearance_v3 import fit_reference,assess,POLICY
from run_prompt_contrast import digest,save,verify


def main():
    files=[Path(__file__).resolve(),Path(__file__).with_name('reference_wire_appearance_v2.py'),
           Path(__file__).with_name('reference_wire_appearance_v3.py')]
    pins={str(p):digest(p) for p in files}
    runner.OUT=runner.ROOT/'artifacts/wire_appearance_reference_v3_20261008'
    runner.fit_reference=fit_reference;runner.assess=assess;runner.POLICY=POLICY
    runner.main();verify(pins)
    protocol=json.loads((runner.OUT/'protocol.json').read_text(encoding='utf-8'))
    protocol['pins'].update(pins);save(runner.OUT/'protocol.json',protocol)


if __name__=='__main__':main()
