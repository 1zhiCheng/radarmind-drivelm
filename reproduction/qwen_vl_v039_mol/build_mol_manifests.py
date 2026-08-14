#!/usr/bin/env python3
"""Build leakage-audited task manifests for the v0.39A MoL pilot."""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

TASKS=("perception","prediction","planning","behavior")
def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--train-jsonl',required=True); p.add_argument('--dev-jsonl',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 train,dev=read(a.train_jsonl),read(a.dev_jsonl); out=Path(a.output_dir); rng=random.Random(a.seed)
 train_scenes={str(x['scene_id']) for x in train}; dev_scenes={str(x['scene_id']) for x in dev}
 if train_scenes&dev_scenes: raise ValueError('train/dev scene leakage')
 by_train=defaultdict(list); by_dev=defaultdict(list)
 for x in train: by_train[str(x['task'])].append(x)
 for x in dev: by_dev[str(x['task'])].append(x)
 if set(by_train)!=set(TASKS) or set(by_dev)!=set(TASKS): raise ValueError('unexpected tasks')
 outputs={}
 for task in TASKS:
  rows=list(by_train[task]); rng.shuffle(rows); path=out/f'train_{task}.jsonl'; write(path,rows); outputs[task]={'train':len(rows),'dev':len(by_dev[task]),'sha256':sha(path)}
 # Round-robin full-data shared control. This preserves every record once while preventing long task blocks.
  write(out/f'dev_{task}.jsonl',by_dev[task])
 shuffled={t:list(by_train[t]) for t in TASKS}
 for t in TASKS:rng.shuffle(shuffled[t])
 shared=[]
 for i in range(max(map(len,shuffled.values()))):
  for t in TASKS:
   if i<len(shuffled[t]):shared.append(shuffled[t][i])
 shared_path=out/'train_shared_round_robin.jsonl'; write(shared_path,shared)
 report={'schema_version':'drivelm-v039a-mol-manifest-v1','seed':a.seed,'train_records':len(train),'dev_records':len(dev),'train_dev_scene_overlap':0,'routing_source':'task key from official QA hierarchy; answer-independent','tasks':outputs,'shared_records':len(shared),'shared_task_counts':dict(Counter(x['task'] for x in shared)),'shared_sha256':sha(shared_path),'source_train_sha256':sha(a.train_jsonl),'source_dev_sha256':sha(a.dev_jsonl)}
 (out/'manifest_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
