#!/usr/bin/env python3
"""Summarize full-protocol v0.39A controls after frozen screening."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def load(p):return json.loads(Path(p).read_text()) if p and Path(p).is_file() else None
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-ds',required=True);p.add_argument('--selection',required=True);p.add_argument('--shared-ds');p.add_argument('--mol-ds');p.add_argument('--output-json',required=True);a=p.parse_args();b=load(a.baseline_ds);sel=load(a.selection)
 def take(d):
  if d is None:return None
  m=d['metrics'];return {'judge_complete':d['judge']['complete'],'judge_completed':d['judge']['completed'],'planning':m['planning_deepseek_100'],'accuracy':m['accuracy'],'language':m['language_combined'],'match':m['match_score_100'],'final':m['drivelm_ds_final']}
 base=take(b);out={'schema_version':'drivelm-v039a-mol-full-summary-v1','baseline':base,'candidates':{},'selection':sel}
 for name,path in [('shared',a.shared_ds),('mol',a.mol_ds)]:
  v=take(load(path));
  if v is not None:out['candidates'][name]={'metrics':v,'delta_vs_baseline':{k:v[k]-base[k] for k in ['planning','accuracy','language','match','final']}}
 mol=out['candidates'].get('mol');shared=out['candidates'].get('shared')
 gates={'mol_was_screened_for_judge':'mol' in sel['judge_candidates'],'mol_judge_complete':bool(mol and mol['metrics']['judge_complete']),'mol_final_above_baseline':bool(mol and mol['metrics']['final']>base['final']),'mol_planning_regression_at_most_0_5':bool(mol and mol['metrics']['planning']>=base['planning']-0.5),'mol_above_shared_if_available':bool(mol and (not shared or mol['metrics']['final']>shared['metrics']['final']))}
 out['preliminary_promotion_gates']=gates;out['preliminary_promoted']=all(gates.values());out['note']='Preliminary only: same-ID common-eligible audit remains mandatory before promotion.';Path(a.output_json).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
