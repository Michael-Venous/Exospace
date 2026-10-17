import unittest
from types import SimpleNamespace
from brain.agent.live_experiment import LiveJSON


class Session:
    def __init__(self,kind):self.kind=kind;self.receives=0;self.responses=[]
    async def send_client_content(self,*,turns,turn_complete):
        self.request_id=turns.parts[-1].text.splitlines()[0].split(': ',1)[1]
    async def send_tool_response(self,**kw):self.responses.append(kw)
    async def receive(self):
        self.receives+=1
        if self.kind=='leftover' and self.receives==1:
            yield SimpleNamespace(tool_call=None)
            return
        fc=SimpleNamespace(id='call',name='propose',args={'request_id':'old' if self.kind=='stale' else self.request_id,'plan':{'actions':[]}})
        yield SimpleNamespace(tool_call=SimpleNamespace(function_calls=[fc,fc] if self.kind=='multiple' else [fc]))


class LiveTests(unittest.IsolatedAsyncioTestCase):
    async def adapter(self,kind):
        ex=object.__new__(LiveJSON);ex.session=Session(kind)
        async def connect():pass
        ex._connect=connect
        return ex

    async def test_previous_turn_complete_is_not_current_proposal(self):
        ex=await self.adapter('leftover')
        self.assertEqual(await ex._generate(b'jpeg','goal'),{'actions':[]})
        self.assertEqual(ex.session.receives,2)

    async def test_old_request_id_cannot_produce_actions(self):
        ex=await self.adapter('stale')
        with self.assertRaisesRegex(ValueError,'Stale'):await ex._generate(b'jpeg','goal')

    async def test_multiple_proposals_are_rejected_atomically(self):
        ex=await self.adapter('multiple')
        with self.assertRaisesRegex(ValueError,'Multiple'):await ex._generate(b'jpeg','goal')


if __name__=='__main__':unittest.main()


class CleanupTests(unittest.TestCase):
    def test_close_failure_does_not_abort_remaining_benchmark_tasks(self):
        import asyncio
        import tempfile
        import threading
        from pathlib import Path
        from unittest.mock import AsyncMock, Mock
        ex=object.__new__(LiveJSON)
        ex.loop=asyncio.new_event_loop();ex.thread=threading.Thread(target=ex.loop.run_forever)
        ex.thread.start()
        ex.context=SimpleNamespace(__aexit__=AsyncMock(side_effect=TimeoutError()))
        ex.client=SimpleNamespace(aio=SimpleNamespace(aclose=AsyncMock()),close=Mock())
        with tempfile.TemporaryDirectory() as folder:
            ex.log_path=Path(folder)/'calls.jsonl'
            ex.close()
            self.assertIn('live_cleanup',ex.log_path.read_text())
        self.assertFalse(ex.thread.is_alive());self.assertTrue(ex.loop.is_closed())
        ex.client.close.assert_called_once()
