# Source this file in a remote shell session to launch the agent on the remote machine.
# This has only been tested with X11; may not work with Wayland.

export DISPLAY=:0.0
touch ~/.Xauthority
