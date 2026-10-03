import copy,unittest
from core_port_supplement_policy import supplement,supported_tiles

def packet():
    merged=[dict(box_xyxy=[100+i*40,100,120+i*40,120],class_id=0,confidence=.99-i*.01,source_tile=0,support_tiles=[0,1]) for i in range(12)]
    raw=[{**p,'source_tile':tile} for p in merged for tile in (0,1)]
    return dict(source_shape=[1000,1000],windows=[[0,0,800,800],[50,50,1000,1000]],merged_predictions=merged,edge_kept_predictions=raw)

class SupplementTests(unittest.TestCase):
    def test_primary_five_add_five(self):
        result=supplement(packet())
        self.assertEqual(len(result['primary']),5);self.assertEqual(len(result['supplementary']),5)
    def test_primary_preserved(self):
        raw=packet();result=supplement(raw)
        self.assertEqual(result['primary'],raw['merged_predictions'][:5])
    def test_single_tile_not_enough(self):
        raw=packet();raw['edge_kept_predictions']=[p for p in raw['edge_kept_predictions'] if p['source_tile']==0]
        self.assertEqual(supplement(raw)['supplementary'],[])
    def test_weak_second_vote_not_enough(self):
        raw=packet()
        for p in raw['edge_kept_predictions']:
            if p['source_tile']==1:p['confidence']=.5
        self.assertEqual(supplement(raw)['supplementary'],[])
    def test_wrong_class_vote_not_enough(self):
        raw=packet()
        for p in raw['edge_kept_predictions']:
            if p['source_tile']==1:p['class_id']=1
        self.assertEqual(supplement(raw)['supplementary'],[])
    def test_duplicate_tile_ids_not_two_votes(self):
        raw=packet()
        for p in raw['edge_kept_predictions']:p['source_tile']=0
        self.assertEqual(supplement(raw)['supplementary'],[])
    def test_strict_additional_threshold(self):
        raw=packet()
        for p in raw['merged_predictions'][5:]:p['confidence']=.75
        self.assertEqual(supplement(raw)['supplementary'],[])
    def test_immutable(self):
        raw=packet();before=copy.deepcopy(raw);supplement(raw);self.assertEqual(raw,before)

if __name__=='__main__':unittest.main()
