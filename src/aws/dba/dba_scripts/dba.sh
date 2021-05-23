#!/bin/bash
#
# Starting point for dba container. This script is responsible for launching
# the REST server but more importantly, the first time this script is run it
# will configure the dba container and load data into the db container. 

DONE_FILE="/configuration_done"

container_is_configured() {
    [ -e "$DONE_FILE" ] && return 0
    return 1
}

configure() {
    :
    #touch "$DONE_FILE"
}

launch_server() {
    exec /application/run.sh
}

main() {
    container_is_configured || configure
    launch_server
}

main
