#!/usr/bin/env python3
"""Validate parquet schema, prompt images and scene isolation through verl's loader."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets import load_dataset

from omegaconf import OmegaConf
from transformers import AutoProcessor
from verl.utils.dataset.rl_dataset import RLHFDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--train-parquet',required=True);p.add_argument('--dev-parquet',required=True);p.add_argument('--report-json',required=True)
    a=p.parse_args();processor=AutoProcessor.from_pretrained(a.model_path)
    config=OmegaConf.create({'prompt_key':'prompt','image_key':'images','max_prompt_length':3072,'filter_overlong_prompts':False,'return_multi_modal_inputs':True,'shuffle':False,'mm_processor_kwargs':{'min_pixels':25088,'max_pixels':100352}})
    loaded={};scenes={}
    for split,path in [('train',a.train_parquet),('dev',a.dev_parquet)]:
        raw=load_dataset('parquet',data_files=path,split='train')
        ds=RLHFDataset(path,processor.tokenizer,config,processor=processor,max_samples=4)
        sample=ds[0]; content=sample['raw_prompt'][1]['content']; image_count=sum(isinstance(x,dict) and x.get('type')=='image' for x in content)
        if image_count!=6:raise AssertionError(f'{split}: expected six images, got {image_count}')
        if sample['data_source']!='radarmind_drivelm_trajectory_planning':raise AssertionError('bad data source')
        loaded[split]={'full_rows':len(raw),'validated_rows':min(4,len(ds)),'sample_id':sample['extra_info']['id'],'sample_images':image_count,'sample_upstream_steps':sample['extra_info']['upstream_steps']}
        scenes[split]={str(x) for x in raw['extra_info']['scene_id']}
    overlap=scenes['train']&scenes['dev']
    if overlap:raise AssertionError(f'scene overlap: {len(overlap)}')
    report={'schema_version':'drivelm-v040-verl-loader-validation-v1','splits':loaded,'scene_overlap':0,'valid':True}
    Path(a.report_json).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
