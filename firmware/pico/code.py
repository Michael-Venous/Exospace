"""Pico main loop: run Pi commands, report USB host attach/detach."""

import supervisor

import devices
from commands import Commands
from link import Link

link = Link()
commands = Commands(
    devices.Keyboard(), devices.AbsMouse(), devices.Touch(),
    lambda: supervisor.runtime.usb_connected, link.pending,
)

was_mounted = None
while True:
    mounted = supervisor.runtime.usb_connected
    if mounted != was_mounted:
        if was_mounted and not mounted:
            try:
                commands.release_all()
            except OSError:
                pass  # the host is gone, so there is nothing left to release
        link.send({"ev": "mounted" if mounted else "unmounted"})
        was_mounted = mounted
    for cmd in link.poll():
        link.send(commands.handle(cmd))
