"""Exercise the overnight one-shot runner's stop gates without models or sleeps."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_feature_adaptation_acceptance as driver

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'full').mkdir();self.write(self.root/'full/progress.json',dict(status='running',pid=99999))
        self.base=patch.object(driver,'BASE',self.root);self.base.start();self.addCleanup(self.base.stop)
        self.work=patch.object(driver,'ROOT',self.root);self.work.start();self.addCleanup(self.work.stop)
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    def status(self):return json.loads((self.root/'continuation_status.json').read_text(encoding='utf-8'))
    def complete_training(self):self.write(self.root/'full/report.json',dict(status='complete'))
    def fake_evaluator(self,rejected=None,error=None):
        def run(args,**kw):
            mode=args[-1];self.modes.append(mode)
            if mode!=error:self.write(self.root/'evaluation'/mode/'report.json',dict(qualifies=mode!=rejected,summary={},lost_current_targets=0))
            return SimpleNamespace(returncode=2 if mode==error else 0)
        self.modes=[];return patch.object(driver.subprocess,'run',side_effect=run)
    def test_failed_training_stops_without_evaluation(self):
        self.write(self.root/'full/progress.json',dict(status='failed',pid=99999,error='gradient_failure'))
        with self.fake_evaluator():driver.main()
        self.assertEqual(self.status()['status'],'training_failed');self.assertEqual(self.modes,[])
    def test_missing_own_process_stops(self):
        with patch('psutil.pid_exists',return_value=False),self.fake_evaluator():driver.main()
        self.assertEqual(self.status()['status'],'training_process_missing');self.assertEqual(self.modes,[])
    def test_duplicate_runner_does_not_replace_status(self):
        self.write(self.root/'continuation_status.json',dict(status='existing'))
        with self.assertRaises(FileExistsError):driver.main()
        self.assertEqual(self.status()['status'],'existing')
    def test_train_rejection_skips_all_holdouts(self):
        self.complete_training()
        with self.fake_evaluator(rejected='train'):driver.main()
        self.assertEqual(self.modes,['train']);self.assertEqual(self.status()['status'],'candidate_rejected')
    def test_inner_rejection_skips_outer(self):
        self.complete_training()
        with self.fake_evaluator(rejected='inner'):driver.main()
        self.assertEqual(self.modes,['train','inner']);self.assertEqual(self.status()['mode'],'inner')
    def test_subprocess_failure_stops(self):
        self.complete_training()
        with self.fake_evaluator(error='train'):driver.main()
        self.assertEqual(self.modes,['train']);self.assertEqual(self.status()['status'],'evaluation_failed')
    def test_all_pass_still_never_deploys(self):
        self.complete_training()
        with self.fake_evaluator():driver.main()
        self.assertEqual(self.modes,['train','inner','outer'])
        self.assertEqual(self.status()['status'],'source_only_accepted_needs_live_reference_and_gui')
        self.assertFalse(self.status()['automatic_deployment'])
    def test_wait_uses_only_expected_training_then_evaluates(self):
        def finish(_):self.complete_training()
        with patch('psutil.pid_exists',return_value=True) as exists,patch.object(driver.time,'sleep',side_effect=finish),self.fake_evaluator():driver.main()
        exists.assert_called_once_with(99999);self.assertEqual(self.modes,['train','inner','outer'])
if __name__=='__main__':unittest.main()
