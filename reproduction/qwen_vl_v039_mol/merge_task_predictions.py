#!/usr/bin/env python3
"""Merge hard-routed task expert predictions with exact reference coverage."""
from __future__ import annotations
import argparse,json
from pathlib import Path
TASKS=("perception","prediction","planning","behavior")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--references-jsonl',required=True);p.add_argument('--prediction-dir',required=True);p.add_argument('--output-json',required=True);p.add_argument('--report-json',required=True);a=p.parse_args()
 refs=[json.loads(x) for x in Path(a.references_jsonl).read_text().splitlines() if x]; byid={str(x['id']):x for x in refs}; pred={}; counts={}
 for task in TASKS:
  rows=json.loads((Path(a.prediction_dir)/f'{task}_predictions.json').read_text());counts[task]=len(rows)
  for row in rows:
   i=str(row['id'])
   if i in pred:raise ValueError(f'duplicate {i}')
   if i not in byid or str(byid[i]['task'])!=task:raise ValueError(f'route mismatch {i}')
   pred[i]={'id':i,'answer':str(row['answer'])}
 missing=set(byid)-set(pred);extra=set(pred)-set(byid)
 if missing or extra:raise ValueError(f'coverage mismatch missing={len(missing)} extra={len(extra)}')
 ordered=[pred[str(x['id'])] for x in refs];out=Path(a.output_json);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(ordered,ensure_ascii=False,indent=2)+'\n')
 report={'schema_version':'drivelm-v039a-hard-route-v1','router':'official QA hierarchy task key; answer-independent','reference_count':len(refs),'prediction_count':len(ordered),'coverage':1.0,'by_expert':counts};Path(a.report_json).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
