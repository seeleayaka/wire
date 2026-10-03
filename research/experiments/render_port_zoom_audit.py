"""Post-inference visualization only; never supplies proposals or parameters."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import read_image

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('image');parser.add_argument('--mode',default='strict');args=parser.parse_args()
    folder=ROOT/args.directory;case=json.loads((folder/(Path(args.image).stem+'_evaluation.json')).read_text(encoding='utf-8'))
    stage='val01' if folder.name=='outer' else 'train01'
    data=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    image=read_image(data/'images'/stage/args.image);h,w=image.shape[:2]
    import cv2
    def box(coords,color,text):
        l,t,r,b=map(round,coords);cv2.rectangle(image,(l,t),(r,b),color,3)
        cv2.putText(image,text,(l,max(t-8,25)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
    baseline=case['selected']['baseline']['all_predictions']
    for row in baseline:box(row['box_xyxy'],(255,180,0),'Base')
    for row in case['selected'][args.mode]['zoom']:box(row['box_xyxy'],(0,140,255),'New %.2f'%row['confidence'])
    for line in (data/'labels'/stage/(Path(args.image).stem+'.txt')).read_text().splitlines():
        cls,cx,cy,bw,bh=map(float,line.split())
        if cls in (3,4):box([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h],(200,200,200),'GT')
    path=folder/(Path(args.image).stem+'_audit.jpg')
    if path.exists():raise FileExistsError(path)
    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,94]);assert ok
    encoded.tofile(str(path));print(str(path))

if __name__=='__main__':main()
