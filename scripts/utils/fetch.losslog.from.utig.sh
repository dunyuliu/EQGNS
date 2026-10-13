model="case4.eq.meshnet"
# <user>@<host> and <remote root> are placeholders; set them for your own
# remote layout (never commit the real values).
scp "${EQGNS_REMOTE_USER_HOST:?set <user>@<host>}:${EQGNS_REMOTE_ROOT:?set the remote gns root}/gns-sample/${model}/models/loss_log.txt" .
