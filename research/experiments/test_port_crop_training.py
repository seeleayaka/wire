import unittest
from inspection_agent.port_crop_training import training_options,select_smoke_records


class TrainingRecipeTests(unittest.TestCase):
    def test_frozen_full_recipe(self):
        options=training_options('data.yaml','new_output','full')
        self.assertEqual((options['epochs'],options['imgsz'],options['warmup_epochs'],options['seed']),(6,960,1.,20260929))
        self.assertEqual((options['batch'],options['workers'],options['device'],options['freeze']),(1,0,'cpu',10))
        self.assertFalse(options['exist_ok']);self.assertEqual(options['patience'],6)

    def test_smoke_differs_only_in_epoch_and_accumulation_count(self):
        full=training_options('data','out','full');smoke=training_options('data','out','smoke')
        self.assertEqual((full['nbs'],smoke['nbs']),(64,4))
        smoke['epochs']=6;smoke['nbs']=64;self.assertEqual(full,smoke)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):training_options('data','out','resume')

    def test_smoke_accumulation_fits_four_batch_fixture(self):
        options=training_options('data','out','smoke')
        self.assertLessEqual(options.get('nbs',64)/options['batch'],4)

    def test_smoke_has_positive_negative_distinct_sources_and_is_order_stable(self):
        records=[{'split':split,'source_image':f'{split}_{i}.JPG','tile_id':t,'label_count':1 if i<3 else 0}
                 for split in ('train','val') for i in range(6) for t in range(2)]
        selected=select_smoke_records(records)
        self.assertEqual(selected,select_smoke_records(list(reversed(records))))
        self.assertEqual(len(selected['train']),4);self.assertEqual(len(selected['val']),2)
        self.assertEqual(sum(r['label_count']>0 for r in selected['train']),2)

    def test_insufficient_examples_fail_closed(self):
        with self.assertRaises(ValueError):select_smoke_records([])


if __name__=='__main__':unittest.main()
