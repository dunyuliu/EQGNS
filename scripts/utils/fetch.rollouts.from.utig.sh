model="case4.40.np.eq.meshnet"
# <user>@<host> and <remote root> are placeholders; set them for your own
# remote layout (never commit the real values).
scp -r "${EQGNS_REMOTE_USER_HOST:?set <user>@<host>}:${EQGNS_REMOTE_ROOT:?set the remote gns root}/gns-sample/${model}/rollouts.nmp15/*" .
