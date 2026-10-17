"""Pico main loop: run commands from each link, report USB host attach/detach."""

import supervisor

import devices
from commands import Commands
from link import open_links

links = open_links()
commands = Commands(
    devices.Keyboard(), devices.AbsMouse(), devices.Touch(),
    lambda: supervisor.runtime.usb_connected, lambda: any(link.pending() for link in links),
)

was_mounted = None
while True:
    mounted = supervisor.runtime.usb_connected
    if mounted != was_mounted:
        # Report first. On unplug there is no host to release keys toward (a new host
        # starts with nothing pressed), and sending reports to a missing host stalls.
        for link in links:
            link.send({"ev": "mounted" if mounted else "unmounted"})
        was_mounted = mounted
    for link in links:
        for cmd in link.poll():
            link.send(commands.handle(cmd))
