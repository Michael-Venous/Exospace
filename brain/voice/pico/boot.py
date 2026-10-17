import usb_cdc

# A second USB serial port carries the brain's commands; the first stays the REPL console.
usb_cdc.enable(console=True, data=True)
