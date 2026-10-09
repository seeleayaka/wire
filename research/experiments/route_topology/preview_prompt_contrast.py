"""Inspect completed cases while the serial fixed protocol continues unchanged."""
import argparse
import json
import sys
from pathlib import Path
from analyze_prompt_contrast import audit_run, contact_sheet, overlaps, OUT
from run_prompt_contrast import digest, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    name=parser.parse_args().case
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    cases=[c for c in protocol['cases'] if c['id']==name]
    if len(cases)!=1:
        raise ValueError('case absent from frozen protocol')
    case=cases[0]
    results={}
    for prompt in protocol['prompts']:
        origin,records,arrays,summary=audit_run(OUT/name/prompt)
        results[prompt]=(origin,records,arrays,summary)
    preview=OUT/name/'preview'
    preview.mkdir(exist_ok=False)
    for prompt,(_,records,arrays,summary) in results.items():
        contact_sheet(case['source']['path'],records,arrays,preview/f'{prompt}_all_masks.png')
        from PIL import Image
        with Image.open(preview/f'{prompt}_all_masks.png') as sheet:
            # Four full rows/page: every instance stays visible at native display scale.
            for page,y in enumerate(range(0,sheet.height,4*345),1):
                sheet.crop((0,y,sheet.width,min(sheet.height,y+4*345))).save(
                    preview/f'{prompt}_page_{page:02d}.png')
        save(preview/f'{prompt}_geometry.json',{'summary':summary,'records':records,
                 'source_binding':case['source'],'port_identity_confirmed':False})
    completed_file=OUT/'inference_report.json'
    all_complete=(completed_file.exists() and json.loads(completed_file.read_text(encoding='utf-8')).get('status')=='complete')
    report={'case':name,'case_inference_complete':True,'overall_inference_complete':all_complete,
            'protocol_sha256':digest(OUT/'protocol.json'),
            'cable':results['cable'][3],'wire':results['wire'][3],
            'cable_to_wire_overlap':overlaps(results['cable'][2],results['wire'][2]),
            'same_model_observer_count':1,'confirmed_connections':0}
    save(preview/'report.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
