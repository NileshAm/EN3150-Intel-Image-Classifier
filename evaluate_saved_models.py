"""Reproduce the final four-model evaluation without running training or searches.

Place the dataset and selected checkpoints as described in README_submission.md,
then run: python evaluate_saved_models.py
The implementation executes the corresponding documented notebook cells directly.
"""
from pathlib import Path
import json
import os
import argparse
import hashlib
import matplotlib

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Check required inputs and checksums without loading images or evaluating models.')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    os.chdir(root)
    matplotlib.use('Agg')
    notebook=json.loads((root/'CNN.ipynb').read_text(encoding='utf-8'))
    cells={cell['id']:cell for cell in notebook['cells']}
    sequence=[
        '13ff55f7','0fcdad3c','sota-step1-checks',
        'sota-step5-setup','sota-step5-helpers',
        'sota-step5-modela','sota-step5-modelb',
        'sota-step5-mnv2','sota-step5-enb0','sota-step5-summary',
    ]
    namespace={'__name__':'__main__','__file__':str(root/'CNN.ipynb')}
    if args.check:
        for cid in sequence:
            compile(''.join(cells[cid]['source']),cid,'exec')
        manifest=json.loads((root/'sota_results/evaluation/run_20261018_070505_412523/evaluation_manifest.json').read_text())
        for name,record in manifest['models'].items():
            checkpoint=root/record['checkpoint'].replace('\\','/')
            assert checkpoint.is_file(),checkpoint
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==record['sha256'],name
            assert (checkpoint.parent/'training_info.json').is_file()
        for model in ['model_a','model_b']:
            assert (root/'sota_results'/model/'latest_completed.json').is_file()
        for split in ['seg_train','seg_test']:
            for label in manifest['labels']:
                assert (root/'dataset'/split/label).is_dir(),(split,label)
        print('Preflight passed: selected cells compile; model files, metadata, registrations and class folders exist; checkpoint checksums match.')
        print('Full data fingerprints are checked during evaluation, not by this preflight.')
        return
    for cid in sequence:
        cell=cells[cid]
        if cell['cell_type']!='code': raise ValueError(cid+' is not a code cell')
        print('\nExecuting evaluation cell:',cid,flush=True)
        exec(compile(''.join(cell['source']),str(root/'CNN.ipynb')+'#'+cid,'exec'),namespace)
    print('\nEvaluation results:',namespace['sota_eval_dir'].resolve())

if __name__=='__main__': main()
