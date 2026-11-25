"""Continue candidate evaluation after an existing trainer; never publish artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]

def run(candidate, csv, old_csv, trainer_pid):
    candidate=Path(candidate).resolve()
    if (ROOT/'models'/'new').resolve() not in candidate.parents:
        raise ValueError('Candidate must be inside models/new.')
    state_path=candidate/'evaluation_progress.json'
    def state(stage, **extra):
        value={'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),'deployed':False,**extra}
        pending=state_path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2),encoding='utf-8');pending.replace(state_path)
        print(stage,flush=True)
    process=psutil.Process(trainer_pid) if psutil.pid_exists(trainer_pid) else None
    state('Waiting for final expert training and validation selection')
    while True:
        manifest=json.loads((candidate/'model_manifest.json').read_text(encoding='utf-8'))
        if manifest['status'].startswith('fitted_and_scored_candidate_only'):break
        if process is None or not process.is_running():
            state('blocked: trainer exited before completing',training_status=manifest['status'])
            return 1
        time.sleep(20)
    jobs=[('paired_comparison','scripts.compare_public_candidate',[csv,str(candidate),old_csv]),
          ('single_guardrails','scripts.evaluate_public_baselines',[csv,str(candidate)]),
          ('flat_ablation_ood','scripts.evaluate_public_studies',[csv,str(candidate)])]
    for stage,module,args in jobs:
        state(stage)
        with (candidate/(stage+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state('failed: '+stage,exit_code=result.returncode,log=stage+'.log')
            return result.returncode
    report=json.loads((candidate/'promotion_comparison.json').read_text(encoding='utf-8'))
    baselines=json.loads((candidate/'baseline_selection.json').read_text(encoding='utf-8'))
    state('Evaluation complete; candidate requires remaining deployment checks' if report['metric_gate_passed'] else 'Evaluation complete; candidate rejected by metric gate',
          metric_gate_passed=report['metric_gate_passed'],checks=report['checks'],
          all_baselines_measured=baselines['measured_baselines']==baselines['required_baselines'])
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('candidate');parser.add_argument('csv');parser.add_argument('old_csv');parser.add_argument('--trainer-pid',type=int,required=True)
    args=parser.parse_args();raise SystemExit(run(args.candidate,args.csv,args.old_csv,args.trainer_pid))
