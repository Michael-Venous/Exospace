# Pico firmware ("hands")

**Owner:** Hardware

CircuitPython firmware that presents as a composite HID device: a boot-protocol keyboard, an absolute mouse (0–32767) and a single-touch digitizer. It runs the newline-delimited JSON UART protocol in [PLAN.md §5.1](../../PLAN.md).

The firmware holds no logic beyond the protocol. On USB `unmounted`, it must `release_all`.

Wiring, Pi setup and the bring-up order: [docs/hardware-bringup.md](../../docs/hardware-bringup.md).
