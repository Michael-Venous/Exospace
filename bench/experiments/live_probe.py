"""Opt-in Gemini Live screenshot-to-function-call probe; no device actions/audio.

This tests API feasibility, not a full voice agent or end-to-end GUI executor.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

from google import genai
from google.genai import types

from brain.browser import BrowserTarget
from brain.config import Config


async def probe(client, image: bytes, expected: str, transport: str) -> dict:
    config=types.LiveConnectConfig(
        response_modalities=['AUDIO'],output_audio_transcription={},
        system_instruction='Read the supplied screen and call report_observation with the exact verification code. '
                           'Do not guess or use earlier screens. Do not perform any actions. Use the function instead of speaking the answer.',
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
            name='report_observation',description='Record the verification code read directly from the current screenshot. This function has no external effects.',
            behavior='BLOCKING',parameters={'type':'OBJECT','properties':{'value':{'type':'STRING'}},'required':['value']})])])
    start=time.monotonic();row={'model':'gemini-3.8-live','transport':transport,'expected':expected,'success':False,'tool_calls':[],
                              'audio_bytes_received':0,'transcript':'','error':''}
    try:
        async with asyncio.timeout(40):
            async with client.aio.live.connect(model=row['model'],config=config) as session:
                row['connect_s']=time.monotonic()-start
                sent=time.monotonic()
                text=types.Part(text='Read the verification code on this screenshot. Call report_observation with that exact code.')
                if transport=='video':
                    await session.send_realtime_input(video=types.Blob(data=image,mime_type='image/jpeg'))
                    parts=[text]
                else:
                    parts=[types.Part.from_bytes(data=image,mime_type='image/jpeg'),text]
                await session.send_client_content(turns=types.Content(role='user',parts=parts),turn_complete=True)
                async for response in session.receive():
                    row.setdefault('first_event_s',time.monotonic()-sent)
                    if response.server_content and response.server_content.output_transcription:
                        row['transcript']+=response.server_content.output_transcription.text or ''
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                row['audio_bytes_received']+=len(part.inline_data.data)
                                row.setdefault('first_audio_s',time.monotonic()-sent)
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            row['tool_calls'].append({'name':fc.name,'args':fc.args,'elapsed_s':time.monotonic()-sent})
                            if fc.name=='report_observation':
                                row['success']=fc.args=={'value':expected}
                                row['tool_call_s']=time.monotonic()-sent
                            await session.send_tool_response(function_responses=types.FunctionResponse(
                                id=fc.id,name=fc.name,response={'recorded':True}))
                        break
    except Exception as exc:
        row['error']=type(exc).__name__
        row['status_code']=getattr(exc,'code',None)
    row['wall_s']=time.monotonic()-start
    return row


def prepare_screens(out, runs):
    images=[];codes=['Q7N42','R8K51','M6P29']
    target=BrowserTarget('about:blank')
    try:
        for i in range(runs):
            code=codes[i]
            target.page.set_content('<style>body{font:32px Arial;padding:60px;background:#eef2f8;color:#18283b}</style>'
                                    f'<h1>Local verification screen</h1><p>Project: Exospace</p><p>Verification code: <b>{code}</b></p><p>Status: ready</p>')
            image=target.page.screenshot(type='jpeg',quality=90)
            (out/f'screen-{i}.jpg').write_bytes(image);images.append(image)
    finally:target.close()
    return images,codes


async def main_async(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Use a fresh output directory')
    images,codes=await asyncio.to_thread(prepare_screens,out,args.runs)
    cfg=Config.load();client=genai.Client(api_key=cfg.gemini_api_key)
    rows=[]
    try:
        for i,image in enumerate(images):
            row=await probe(client,image,codes[i],args.transport);rows.append(row)
            (out/'report.json').write_text(json.dumps({'scope':'Screenshot + typed request -> inert function call; no microphone, speaker playback or HID.',
                                                      'rows':rows},indent=2))
            print(json.dumps(row),flush=True)
    finally:await client.aio.aclose();client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--output',default='work/live-probe-v1')
    parser.add_argument('--transport',choices=['video','content'],default='content');args=parser.parse_args()
    if not 1<=args.runs<=3:parser.error('runs must be 1-3')
    asyncio.run(main_async(args))


if __name__=='__main__':main()
