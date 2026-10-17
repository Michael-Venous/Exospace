"""Experimental Live function-call transport. Never writes HID or plays audio."""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
import json
import threading
import time
import uuid

from google import genai
from google.genai import types


class LiveJSON:
    def __init__(self, cfg, cancel, system, schema):
        self.cancel, self.log_path = cancel, cfg.model_log
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.system, self.schema = system, schema
        self.session = None
        self.context = None
        self.connect_s = 0.0
        self.calls = 0

    def _wait(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            while True:
                self.cancel.check()
                try:
                    return future.result(timeout=.02)
                except FutureTimeout:
                    if future.done():
                        return future.result()
        finally:
            if not future.done():
                future.cancel()

    async def _connect(self):
        if self.session is None:
            start = time.monotonic()
            config = types.LiveConnectConfig(response_modalities=['AUDIO'],
                system_instruction=self.system+'\nAlways call propose exactly once with request_id from the latest message. This proposes a plan; it does NOT execute it. Do not speak. Use only the latest screenshot for current state.',
                tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                    name='propose', description='Propose a bounded plan for validation; no actions execute here.', behavior='BLOCKING',
                    parameters_json_schema={'type':'object','properties':{
                        'request_id':{'type':'string'},'plan':self.schema},'required':['request_id','plan']})])])
            self.context = self.client.aio.live.connect(model='gemini-3.8-live', config=config)
            self.session = await self.context.__aenter__()
            self.connect_s += time.monotonic()-start

    def warm(self):
        self._wait(self._warm())

    async def _warm(self):
        async with asyncio.timeout(25):
            await self._connect()

    async def _generate(self, image, prompt):
        async with asyncio.timeout(25):
            await self._connect()
            request_id = uuid.uuid4().hex
            await self.session.send_client_content(turns=types.Content(role='user',parts=[
                types.Part.from_bytes(data=image,mime_type='image/jpeg'),
                types.Part(text=f'Request ID: {request_id}\n'+prompt)]),turn_complete=True)
            while True:
                async for response in self.session.receive():
                    if response.tool_call:
                        calls = response.tool_call.function_calls
                        for fc in calls:
                            await self.session.send_tool_response(function_responses=types.FunctionResponse(
                                id=fc.id,name=fc.name,response={'received_for_validation':True,'executed':False}))
                        if len(calls)!=1:
                            raise ValueError('Multiple Live proposals')
                        fc=calls[0]
                        if fc.name!='propose' or fc.args.get('request_id')!=request_id:
                            raise ValueError('Stale or unexpected Live proposal')
                        if not isinstance(fc.args.get('plan'),dict):
                            raise ValueError('Invalid Live plan')
                        return fc.args['plan']
                # A previous tool acknowledgement can leave a turn_complete in
                # the stream. Only a matching request ID satisfies this request;
                # keep receiving within the bounded deadline.
                await asyncio.sleep(0)

    def generate(self, image, prompt, system, schema, *, role, run_id='', step=0):
        start=time.monotonic();success=False;error=''
        try:
            result=self._wait(self._generate(image,prompt))
            self.calls+=1;success=True
            return result
        except Exception as exc:
            error=type(exc).__name__
            raise RuntimeError('Live proposal failed: '+error) from None
        finally:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as f:
                f.write(json.dumps({'run_id':run_id,'step':step,'role':'live','model':'gemini-3.8-live',
                    'elapsed_s':time.monotonic()-start,'success':success,'error':error})+'\n')

    async def _close(self):
        errors=[]
        try:
            async with asyncio.timeout(3):
                if self.context is not None:
                    await self.context.__aexit__(None,None,None)
        except Exception as exc:
            errors.append(type(exc).__name__)
        try:
            async with asyncio.timeout(1):
                await self.client.aio.aclose()
        except Exception as exc:
            errors.append(type(exc).__name__)
        # This loop belongs only to this adapter. Stop transport keepalives and
        # pending session cleanup before stopping its thread.
        pending=[task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:task.cancel()
        if pending:
            await asyncio.wait(pending,timeout=1)
        if errors:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as f:f.write(json.dumps({'role':'live_cleanup','errors':errors})+'\n')

    def close(self):
        future=asyncio.run_coroutine_threadsafe(self._close(),self.loop)
        try:
            future.result(timeout=6)
        except FutureTimeout:
            future.cancel()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)
            if not self.thread.is_alive():self.loop.close()
            self.client.close()
