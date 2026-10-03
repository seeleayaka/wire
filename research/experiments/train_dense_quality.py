"""Reuse verified fixed plain runner, explicitly replace objective/provenance."""
from pathlib import Path
import train_dense_pixel_plain as runner
from dense_quality_objective import loss


def main():
    paths=[Path(__file__),Path(__file__).with_name('dense_quality_objective.py'),
           runner.ROOT/'artifacts/dense_quality_preregistration_20261003/PLAN.md']
    pins={str(p):runner.sha(p) for p in paths}
    old=(runner.BASE,runner.loss,runner.save)
    def save(path,value):
        assert {p:runner.sha(Path(p)) for p in pins}==pins
        if isinstance(value,dict):value=dict(value)
        if path.name=='protocol.json':
            value['pins']=dict(value['pins'],**pins);value.pop('only_difference_from_D4_continuation',None)
            value.update(objective='QFL soft predicted-IoU target at GT centers plus smoothL1 and GIoU',
                         quality_targets_training_only=True,no_GT_in_inference=True,not_full_GFL_reproduction=True)
        if path.name=='report.json':value.update(joint_localization_quality_training=True,quality_code_sha256=pins[str(paths[1])])
        return old[2](path,value)
    try:
        runner.BASE=runner.ROOT/'artifacts/dense_quality_20261003';runner.loss=loss;runner.save=save;runner.main()
    finally:runner.BASE,runner.loss,runner.save=old


if __name__=='__main__':main()
