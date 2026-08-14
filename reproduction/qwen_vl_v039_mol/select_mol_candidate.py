#!/usr/bin/env python3
"""Freeze the no-judge v0.39A screen for M01 shared and M10 hard-routed MoL."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def flat(off,st):
 return {'coverage':off['coverage'],'exact_match':off['overall']['exact_match'],'token_f1':off['overall']['token_f1'],'rouge_l':off['overall']['rouge_l'],'planning_token_f1':off['by_task']['planning']['token_f1'],'mc_accuracy':off['multiple_choice']['accuracy'],'eligible_count':st['eligible_count'],'anchor_coordinate_f1':st['anchor_coordinate_macro']['f1'],'tag3_coordinate_f1':st['tag3_coordinate_macro']['f1']}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['baseline-offline','baseline-structural','shared-offline','shared-structural','mol-offline','mol-structural','output-json']:p.add_argument('--'+n,required=True)
 a=p.parse_args(); b=flat(load(a.baseline_offline),load(a.baseline_structural)); out={'schema_version':'drivelm-v039a-mol-offline-selection-v1','baseline':b,'candidates':{}}
 for name in ('shared','mol'):
  c=flat(load(getattr(a,name+'_offline')),load(getattr(a,name+'_structural')));d={k:c[k]-b[k] for k in c};g={'coverage_100_percent':c['coverage']==1.0,'token_f1_strictly_improved':d['token_f1']>0,'planning_token_f1_within_0_2pp':d['planning_token_f1']>=-0.002,'mc_within_0_2pp':d['mc_accuracy']>=-0.002,'eligible_within_20':d['eligible_count']>=-20,'anchor_f1_within_0_2pp':d['anchor_coordinate_f1']>=-0.002};out['candidates'][name]={'metrics':c,'delta':d,'screening_gates':g,'eligible_for_judge':all(g.values())}
 eligible=[n for n,v in out['candidates'].items() if v['eligible_for_judge']]
 eligible.sort(key=lambda n:(out['candidates'][n]['metrics']['token_f1'],out['candidates'][n]['metrics']['planning_token_f1'],out['candidates'][n]['metrics']['mc_accuracy']),reverse=True)
 out['judge_candidates']=eligible;out['policy']='No semantic judge is visible to this screen. Screening is not promotion; full DriveLM-DS and same-ID gates remain mandatory.';Path(a.output_json).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
