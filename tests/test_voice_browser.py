"""Browser regressions for speech session lifecycle and ordered voice commands."""

import importlib.util
import queue
import unittest
from unittest.mock import Mock

from brain.voice.layer import page_server


@unittest.skipUnless(importlib.util.find_spec("playwright"), "requires the browser extra")
class VoiceBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import Error, sync_playwright
        playwright = sync_playwright().start()
        cls.addClassCleanup(playwright.stop)
        try:
            cls.browser = playwright.chromium.launch(args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            ])
        except Error as exc:
            raise unittest.SkipTest(f"no Chromium to drive: {exc}")
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.agent = Mock()
        self.agent.screen.return_value = None
        server = page_server(self.agent, 0, 8781, queue.Queue())
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.page = self.browser.new_page(reduced_motion="reduce")
        self.addCleanup(self.page.close)
        self.page.add_init_script("""
            window.events = [];
            window.EventSource = class { constructor() { events.push(this); } };
            window.spoken = [];
            speechSynthesis.speak = utterance => spoken.push(utterance.text);
            window.sockets = [];
            window.WebSocket = class {
                constructor() {
                    this.readyState = 0; this.sent = []; sockets.push(this);
                    setTimeout(() => { this.readyState = 1; this.onopen?.(); }, 0);
                }
                send(data) { this.sent.push(data); }
                close(code = 1000) { this.readyState = 3; this.onclose?.({code}); }
                result(text) {
                    this.onmessage({data: JSON.stringify({type: 'Results', is_final: true,
                        speech_final: true, channel: {alternatives: [{transcript: text}]}})});
                }
            };
        """)
        self.page.goto(f"http://127.0.0.1:{server.server_address[1]}/")

    def test_server_voiced_lines_are_not_spoken_twice(self):
        self.page.evaluate("events[0].onmessage({data: JSON.stringify({say: 'Done.', voiced: true})})")
        self.assertEqual(self.page.locator("#says").text_content(), "Done.")
        self.assertEqual(self.page.evaluate("spoken"), [])
        self.page.evaluate("events[0].onmessage({data: JSON.stringify({say: 'Continue?'})})")
        self.assertEqual(self.page.evaluate("spoken"), ["Continue?"])

    def listen(self):
        self.page.click("#talk")
        self.page.wait_for_function("sockets.length === 1 && sockets[0].readyState === 1 && !starting")

    def test_a_tap_while_the_previous_stream_drains_starts_the_next_session(self):
        self.listen()
        self.page.evaluate("release()")
        self.page.wait_for_function("sockets[0].sent.includes(JSON.stringify({type: 'CloseStream'}))")
        self.page.click("#talk")
        self.page.evaluate("sockets[0].close()")
        self.page.wait_for_function("sockets.length === 2 && !starting", timeout=2000)
        self.assertEqual(self.page.locator("#talk").get_attribute("aria-pressed"), "true")

    def test_service_retries_once_without_an_extra_barge_in(self):
        self.listen()
        self.page.evaluate("sockets[0].close(1011)")
        self.page.wait_for_function("sockets.length === 2 && !starting")
        self.page.evaluate("sockets[1].close(1011)")
        self.page.wait_for_function("!held && !stopListening")
        self.assertEqual(self.page.evaluate("sockets.length"), 2)
        self.assertEqual(self.agent.interrupt.call_count, 1)
        self.page.click("#talk")
        self.page.wait_for_function("sockets.length === 3 && !starting")

    def test_stalled_stream_cannot_block_the_next_press(self):
        self.listen()
        self.page.evaluate("release()")
        self.page.wait_for_function("sockets[0].sent.includes(JSON.stringify({type: 'CloseStream'}))")
        self.page.click("#talk")
        self.page.wait_for_function("sockets.length === 2 && !starting", timeout=4500)
        self.assertEqual(self.page.locator("#talk").get_attribute("aria-pressed"), "true")

    def test_changing_microphone_waits_for_another_press(self):
        self.listen()
        self.page.evaluate("mics.onchange()")
        self.page.wait_for_function("sockets[0].sent.includes(JSON.stringify({type: 'CloseStream'}))")
        self.page.evaluate("sockets[0].close()")
        self.assertFalse(self.page.evaluate("held"))
        self.assertEqual(self.page.evaluate("sockets.length"), 1)
        self.page.click("#talk")
        self.page.wait_for_function("sockets.length === 2 && !starting")

    def test_commands_wait_for_the_previous_response(self):
        self.page.evaluate("""() => {
            window.requests = []; window.finishRequest = null;
            window.fetch = (path) => {
                requests.push(path);
                if (path === '/talk') return new Promise(resolve => {
                    finishRequest = () => resolve({json: async () => ({ok: true})});
                });
                return Promise.resolve({json: async () => ({ok: true})});
            };
            post('/talk'); send('open browser', false); send('search cats', true);
        }""")
        self.assertEqual(self.page.evaluate("requests"), ["/talk"])
        self.page.evaluate("finishRequest()")
        self.page.wait_for_function("requests.length === 3")
        self.assertEqual(self.page.evaluate("requests"), ["/talk", "/segment", "/segment"])

    def test_final_clauses_are_sent_while_the_microphone_is_still_recording(self):
        self.listen()
        self.page.evaluate("""() => {
            window.segments = [];
            window.fetch = (path, options) => {
                if (path === '/segment') segments.push(JSON.parse(options.body));
                return Promise.resolve({json: async () => ({ok: true})});
            };
            sockets[0].result('open Firefox');
            sockets[0].result('and search for cats');
        }""")
        self.page.wait_for_function("segments.length === 2")
        self.assertTrue(self.page.evaluate("held && !!stopListening"))
        self.assertEqual(self.page.evaluate("segments"), [
            {"text": "open Firefox", "continues": False},
            {"text": "and search for cats", "continues": True},
        ])
        self.assertFalse(self.page.evaluate("sockets[0].sent.includes(JSON.stringify({type: 'CloseStream'}))"))
