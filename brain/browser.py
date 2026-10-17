"""Isolated browser development target: screenshots in, pixel HID commands out.

No DOM or accessibility data enters the agent. Browser automation is only the
capture/input transport, standing in for the missing capture card and Pico.
"""
from __future__ import annotations

import cv2
import numpy as np
import os

from brain.hid import HIDError, HID_MAX


class BrowserTarget:
    def __init__(self, url: str, *, headed: bool = False, width: int = 1280, height: int = 800):
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        try:
            self.browser = self._playwright.chromium.launch(
                headless=not headed, executable_path=os.environ.get('BROWSER_EXECUTABLE') or None)
            self.context = self.browser.new_context(viewport={'width':width,'height':height},
                                                    device_scale_factor=1)
            self.page = self.context.new_page()
            self.page.goto(url, wait_until='load', timeout=15000)
        except BaseException:
            self._playwright.stop()
            raise
        self.width, self.height = width, height
        self._open = True
        self._keymap = {'CTRL':'Control','SHIFT':'Shift','ALT':'Alt','GUI':'Meta',
                       'ENTER':'Enter','ESC':'Escape','TAB':'Tab','BACKSPACE':'Backspace',
                       'DELETE':'Delete','SPACE':'Space','UP':'ArrowUp','DOWN':'ArrowDown',
                       'LEFT':'ArrowLeft','RIGHT':'ArrowRight','HOME':'Home','END':'End',
                       'PAGEUP':'PageUp','PAGEDOWN':'PageDown'}

    def grab(self):
        raw = self.page.screenshot(type='png', animations='disabled', caret='hide')
        return cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)

    def signal_present(self):
        # The monitor thread must not call Playwright's thread-bound APIs.
        return self._open

    def is_mounted(self):
        return self._open

    def on_event(self, callback):
        pass

    def _px(self,x,y):
        return x/HID_MAX*(self.width-1),y/HID_MAX*(self.height-1)

    def send(self, cmd):
        op = cmd['op']
        if op == 'ping':
            return
        if op == 'key':
            self.page.keyboard.press('+'.join(self._keymap.get(k,k.lower()) for k in cmd['keys']))
        elif op == 'type':
            self.page.keyboard.insert_text(cmd['text'])
        elif op in {'click','tap'}:
            self.page.mouse.click(*self._px(cmd['x'],cmd['y']),button=cmd.get('button','left'),
                                  click_count=cmd.get('count',1))
        elif op == 'move':
            self.page.mouse.move(*self._px(cmd['x'],cmd['y']))
        elif op == 'scroll':
            self.page.mouse.wheel(0,cmd['dy']*100)
        elif op == 'swipe':
            self.page.mouse.move(*self._px(cmd['x1'],cmd['y1']))
            self.page.mouse.down()
            self.page.mouse.move(*self._px(cmd['x2'],cmd['y2']),steps=10)
            self.page.mouse.up()
        elif op == 'release_all':
            for key in ('Control','Shift','Alt','Meta'):
                self.page.keyboard.up(key)
            self.page.mouse.up()
        else:
            raise HIDError(f'Unsupported browser input: {op}')

    def close(self):
        self._open = False
        self.context.close()
        self.browser.close()
        self._playwright.stop()
