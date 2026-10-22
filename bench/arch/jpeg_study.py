"""Read-only JPEG quality/grounding comparison; never sends input."""
import json,time,statistics
from pathlib import Path
from unittest.mock import patch
import cv2
from bench.arch.fixtures import render_all
from brain.agent.jev import JevVision
import brain.agent.jev as module
from brain.cancel import CancelToken
from brain.config import Config
from brain.telemetry import Telemetry

OUT=Path('work/jpeg-study')
CASES=[('desktop','chrome','Google Chrome desktop icon'),
       ('inbox','reply','Reply button'),
       ('youtube','search','YouTube search field'),
       ('blender','render_menu','Render menu'),
       ('blender','scripting_ws','Scripting workspace tab'),
       ('phone','settings','Settings app icon'),
       ('desktop',None,'Airplane mode toggle')]

def main():
    boxes=render_all(OUT); cfg=Config.load(); jev=JevVision(cfg.typesafe_api_key,CancelToken(),OUT/'calls.jsonl',model=cfg.jev_model)
    original_encode=cv2.imencode; original_prepare=module.prepare_image; rows=[]
    try:
        for repeat in range(2):
            for screen,target,description in CASES:
                if target is not None and target not in boxes[screen]: raise ValueError((screen,target,boxes[screen].keys()))
                jpeg=(OUT/f'{screen}.jpg').read_bytes()
                for quality in ((90,70) if repeat==0 else (70,90)):
                    encoded_size=[]
                    def prepare(data):
                        with patch.object(cv2,'imencode',side_effect=lambda ext,img,params: original_encode(ext,img,[cv2.IMWRITE_JPEG_QUALITY,quality])):
                            encoded,size,padded=original_prepare(data)
                        encoded_size.append(len(encoded)); return encoded,size,padded
                    telemetry=Telemetry(Path('bench/results.csv'),f'jpeg-quality-{quality}'); run=telemetry.new_run(); start=time.monotonic()
                    with patch.object(module,'prepare_image',side_effect=prepare): point=jev.locate(jpeg,description,run_id=run,step=1)
                    elapsed=time.monotonic()-start
                    correct=point is None if target is None else point is not None and boxes[screen][target][0]<=point[0]<=boxes[screen][target][2] and boxes[screen][target][1]<=point[1]<=boxes[screen][target][3]
                    row=dict(repeat=repeat,screen=screen,target=target,quality=quality,bytes=encoded_size[0],point=point,correct=correct,seconds=elapsed)
                    rows.append(row); (OUT/'results.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(row),flush=True)
                    telemetry.step(run,'synthetic',f'{screen}-{target}',1,t_capture=0,t_upload=0,t_model=elapsed,t_execute=0,t_settle=0)
                    telemetry.finish(run,'synthetic',f'{screen}-{target}',1,correct,elapsed)
    finally: jev.close()
    for quality in (90,70):
        rs=[r for r in rows if r['quality']==quality]
        print(quality,'correct',sum(r['correct'] for r in rs),'/',len(rs),'bytes',sum(r['bytes'] for r in rs),'median_s',statistics.median(r['seconds'] for r in rs),flush=True)
if __name__=='__main__': main()
