import copy
import unittest
from core_port_zoom_policy import proposals,windows,select_zoom,confirmed

def p(score=.2,box=None,cls=1):
    return dict(confidence=score,class_id=cls,box_xyxy=box or [1000,1000,1020,1040],source_tile=0,support_tiles=[0])
def raw(rows=None):return dict(source_shape=[2736,3648],merged_predictions=rows or [],edge_kept_predictions=rows or [],windows=[])
def case():
    seed=p();a=p(.85);b=p(.8)
    return dict(predictions=raw([seed]),zoom_evidence=[dict(proposal=seed,windows=windows(seed,[2736,3648]),views=[[a],[b]])])

class ZoomTests(unittest.TestCase):
    def test_image_only_proposal_limit(self):
        rows=[p(.2,[200+i*200,800,220+i*200,840]) for i in range(9)]
        self.assertEqual(len(proposals(raw(rows))),6)
    def test_small_low_score_interior_only(self):
        self.assertEqual(len(proposals(raw([p(.2),p(.6,[1800,1000,1820,1040]),p(.2,[2,100,22,140]),p(.2,[300,100,500,140])]))),1)
    def test_duplicate_proposal(self):self.assertEqual(len(proposals(raw([p(),p()]))),1)
    def test_strict_confirm(self):self.assertEqual(len(confirmed(case(),'strict')),1)
    def test_score_modes(self):
        c=case();c['zoom_evidence'][0]['views'][1][0]['confidence']=.6
        self.assertEqual(len(confirmed(c,'strict')),0);self.assertEqual(len(confirmed(c,'standard')),1)
    def test_class_and_agreement_required(self):
        c=case();c['zoom_evidence'][0]['views'][1][0]['class_id']=0
        self.assertEqual(confirmed(c,'strict'),[])
        c=case();c['zoom_evidence'][0]['views'][1][0]['box_xyxy']=[1100,1100,1120,1140]
        self.assertEqual(confirmed(c,'strict'),[])
    def test_identical_views_not_votes(self):
        c=case();c['zoom_evidence'][0]['windows'][1]=c['zoom_evidence'][0]['windows'][0]
        self.assertEqual(confirmed(c,'strict'),[])
    def test_baseline_unchanged_and_no_mutation(self):
        c=case();before=copy.deepcopy(c);chosen=select_zoom(c,'strict')
        self.assertEqual(c,before);self.assertEqual(len(chosen['zoom']),1);self.assertEqual(chosen['primary'],[])
    def test_empty_and_invalid_mode(self):
        self.assertEqual(proposals(raw()),[])
        with self.assertRaises(ValueError):confirmed(case(),'bad')
    def test_windows_stay_on_image(self):
        for w in windows(p(box=[17,17,22,35]),[2736,3648]):
            self.assertTrue(w[0]>=0 and w[1]>=0 and w[2]<=3648 and w[3]<=2736)
    def test_existing_ten_cues_preserved_no_extra_budget(self):
        c=case();rows=[p(.95,[200+i*220,1500,220+i*220,1540]) for i in range(10)]
        votes=[]
        for row in rows:
            for tile in (0,1):
                q=copy.deepcopy(row);q['source_tile']=tile;votes.append(q)
        c['predictions']=raw(rows+[p()]);c['predictions']['edge_kept_predictions']=votes
        chosen=select_zoom(c,'strict')
        self.assertEqual(len(chosen['primary']),5);self.assertEqual(len(chosen['supplementary']),5)
        self.assertEqual(chosen['zoom'],[]);self.assertEqual(len(chosen['all_predictions']),10)
    def test_new_crop_frame_clipped_box_refused(self):
        c=case();c['zoom_evidence'][0]['views'][0][0]['box_xyxy']=[2,1000,1020,1040]
        self.assertEqual(confirmed(c,'standard'),[])
    def test_context_keeps_training_scale_and_zoom_policy_immutable(self):
        from core_port_context_policy import windows as cw,POLICY as cp
        from core_port_zoom_policy import POLICY as zp
        self.assertEqual(zp['crop_size'],640);self.assertEqual(cp['crop_size'],1280)
        a,b=cw(p(),[2736,3648]);self.assertNotEqual(a,b)
        self.assertEqual(a[2]-a[0],1280);self.assertEqual(a[3]-a[1],1280)
    def test_recenter_confirms_high_score_single_view_backlog(self):
        from core_port_recenter_policy import proposals as rp
        rows=[p(.9,[200+i*220,1500,220+i*220,1540]) for i in range(7)]
        self.assertEqual(len(proposals(raw(rows))),0)
        self.assertEqual(len(rp(raw(rows))),2)
    def test_fast_recheck_is_strong_prefix_only(self):
        from core_port_recheck_policy import proposals as fp
        from core_port_recenter_policy import proposals as rp
        rows=[p(.9,[200+i*220,1500,220+i*220,1540]) for i in range(7)]+[p(.2)]
        self.assertEqual(fp(raw(rows)),[q for q in rp(raw(rows)) if q['confidence']>.5])
        self.assertEqual(fp(raw([p(.5)])),[])

if __name__=='__main__':unittest.main()
