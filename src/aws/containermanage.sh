#!/bin/bash

#
# This script manages 3 containers:
# - Frontend web application
# - Database
# - Container for interacting with and populating the database
#

UI_CONTAINER="ui_application"   # aka "ui"
DB_CONTAINER="shared_postgres"  # aka "db"
DBA_CONTAINER="dba"             # aka "dba"

# The DB container gets a persistent volume
# I haven't validated adding data and then destroying+creating the DB container
DB_VOLUME="postgres_volume"
DB_MOUNT_LOC="/var/lib/postgresql/data" # directory within container

# Models path to be mounted inside dba container
DBA_MODELS="/home/ec2-user/260_capstone/models"

UI_ARCHIVE="ui.tar.gz"
#DB_ARCHIVE="db.tar.gz"
DB_ARCHIVE="db.sql.gz"
DBA_ARCHIVE="dba.tar.gz"

# User may be asked to fill these out when running the script.
DBPW_DEFUALT="vulnerwatch"
DBPW=""
DOCKERCLEAR=""

# CLI arguments control:
MODE=""
CONNECTTO=""
EXPORTDIR=""
IMPORTDIR=""
STOPCONT=""
STARTCONT=""
DELETECONT=""
RUNCONT=""
BUILDCONT=""

###############################################################################
# Functions
###############################################################################

err() {
    echo -e "ERROR: $@" >&2
}

fatal() {
    echo -e "FATAL: $@" >&2
    exit 1
}

# Convert shorthand/"external" strings to internally used, longer strings.
ext_to_int() {
    case "$1" in
        "ui") echo "$UI_CONTAINER";;
        "db") echo "$DB_CONTAINER";;
        "dba") echo "$DBA_CONTAINER";;
        "*") echo "This should never happen" >&2;;
    esac
}

usage() {
    echo -e "Usage: $0 [fresh|start|stop|delete]|[connect ui|db|dba]|[export|import <dir>]"
    echo -e "fresh:    delete our docker entities, generate new ones from scratch, and start them"
    echo -e "start:    starts our docker containers (can specify individual containers)"
    echo -e "stop:     stops our docker containers (can specify individual containers)"
    echo -e "delete:   deletes our docker images and containers (can specify individual containers)"
    echo -e "connect:  launch shell on specified (running) container"
    echo -e "import:   imports and starts our docker images from specified directory"
    echo -e "export:   exports our docker images to specified directory"
}

# Sets a few global variables
getargs() {
    MODE="$1"

    case "$1" in
        "fresh")  DOCKERCLEAR="y";;
        "start")  STARTCONT=$(ext_to_int "$2");;
        "stop")   STOPCONT=$(ext_to_int "$2");;
        "delete") DELETECONT=$(ext_to_int "$2"); DOCKERCLEAR="y";;
        "connect")
            CONNECTTO="$2"
            if [[ "$CONNECTTO" != "ui" ]] && [[ "$CONNECTTO" != "db" ]] && [[ "$CONNECTTO" != "dba" ]]; then
                err "Invalid 'connect' argument: '$CONNECTTO'. Must be one of [ui, db, dba].\n"
                usage
                exit 1
            fi
            ;;

        "export")
            EXPORTDIR="$2"
            if [ -z "$EXPORTDIR" ]; then
                fatal "'export' argument requires a directory to export to."
            fi
            ;;

        "import")
            IMPORTDIR="$2"
            if [ -z "$IMPORTDIR" ]; then
                fatal "'import' argument requires a directory to import from."
            fi
            ;;

        # Hidden options. They can be added to the usage if the difference
        # between "run" and "start" can be adequately explained.
        #
        # Docker "run" commands only need to happen after a docker image was
        # built, then "start" should be used to start a not-running container.
        "build")  BUILDCONT=$(ext_to_int "$2");;
        "run")  RUNCONT=$(ext_to_int "$2");;

        *) usage; exit 1;;
    esac

}

# Meant for verifying command exit status
verify_es() {
    if [[ $1 -ne 0 ]]; then
        exit 1
    fi
}

# Gets all the user input we'll need
get_input() {
    if [[ "$MODE" == "fresh" ]] || [[ "$MODE" == "import" ]]; then
        read -r -p "Enter desired postgres password (leave blank to use default): " -s DBPW
        echo
        if [[ $DBPW == "" ]]; then
            DBPW="$DBPW_DEFUALT"
        fi
    fi
}

install_dependencies() {
    pkgs=(
        'docker'
    )
    yum install -y ${pkgs}
}

start_services() {
    service docker start 2>/dev/null
}

clear_entities() {
    docker container stop "$1" 2>/dev/null
    docker container rm "$1"
    docker rmi "$1"
}

clear_out_docker() {
    for cont in "$UI_CONTAINER" "$DB_CONTAINER" "$DBA_CONTAINER"; do
        clear_entities "$cont" 2>/dev/null
    done
}

completely_clear_out_docker() {
    docker container stop $(docker ps -aq) 2>/dev/null
    docker container rm $(docker container ls -aq) 2>/dev/null
    docker rmi $(docker images -aq)
}

setup() {
    [[ "$MODE" == "fresh" ]] && install_dependencies

    start_services
}

###############################################################################
# Docker functions
###############################################################################

image_exists() {
    docker image ls | grep -q "$1" && return 0
    return 1
}

container_exists() {
    docker container list -a | grep -q "$1" && return 0
    return 1
}

container_is_running() {
    docker ps | grep -q "$1" && return 0
    return 1
}

ui_image_exists() { return $(image_exists "$UI_CONTAINER"); }
db_image_exists() { return $(image_exists "$DB_CONTAINER"); }
dba_image_exists() { return $(image_exists "$DBA_CONTAINER"); }

ui_container_exists() { return $(container_exists "$UI_CONTAINER"); }
db_container_exists() { return $(container_exists "$DB_CONTAINER"); }
dba_container_exists() { return $(container_exists "$DBA_CONTAINER"); }

ui_container_is_running() { return $(container_is_running "$UI_CONTAINER"); }
db_container_is_running() { return $(container_is_running "$DB_CONTAINER"); }
dba_container_is_running() { return $(container_is_running "$DBA_CONTAINER"); }

build_ui_container() {
    ui_container_exists && return

    cd ui || exit 1

    docker build -t $UI_CONTAINER .
    es=$?
    verify_es $es

    cd ..
}

build_dba_container() {
    dba_container_exists && return

    cd dba || exit 1

    docker build -t $DBA_CONTAINER .
    es=$?
    verify_es $es

    cd ..
}

build_container() {
    case "$1" in
        "$UI_CONTAINER") build_ui_container;;
        "$DBA_CONTAINER") build_dba_container;;
    esac
}

build_containers() {
    build_ui_container
    build_dba_container
}

run_ui_container() {
    # "--net host" is what allows this container to access the db container.
    docker run --name "$UI_CONTAINER" -p 80:80 -d --net host "$UI_CONTAINER"
}

run_db_container() {
    docker volume create "$DB_VOLUME" 2>/dev/null

    # If the order of the below options is changed, the postgres container may
    # not be publicly available.
    docker run --name "$DB_CONTAINER" \
               -d \
               --mount source="$DB_VOLUME",target="$DB_MOUNT_LOC" \
               -e POSTGRES_PASSWORD="$DBPW" \
               -p 5432:5432 \
               postgres:latest
}

run_dba_container() {
    # Use the following command if GPU is available:
    # docker run --name "$DBA_CONTAINER" -p 81:81 -d --gpus all -v $DBA_MODELS:/models --net host "$DBA_CONTAINER"
    docker run --name "$DBA_CONTAINER" -p 81:81 -d -v $DBA_MODELS:/models --net host "$DBA_CONTAINER"
}

run_container() {
    case "$1" in
        "$UI_CONTAINER") ui_container_is_running || run_ui_container;;
        "$DB_CONTAINER") db_container_is_running || run_db_container;;
        "$DBA_CONTAINER") dba_container_is_running || run_dba_container;;
    esac
}

run_containers() {
    ui_container_is_running || run_ui_container
    db_container_is_running || run_db_container
    dba_container_is_running || run_dba_container
}

start_container() {
    docker container start $1
}

start_containers() {
    for cont in "$UI_CONTAINER" "$DB_CONTAINER" "$DBA_CONTAINER"; do
        container_is_running "$cont" || start_container "$cont"
    done
}

stop_container() {
    docker container stop $1
}

stop_containers() {
    for cont in "$UI_CONTAINER" "$DB_CONTAINER" "$DBA_CONTAINER"; do
        container_is_running "$cont" && stop_container "$cont"
    done
}

connect_container() {
    case "$1" in
        "ui") container=$UI_CONTAINER;;
        "db") container=$DB_CONTAINER;;
        "dba") container=$DBA_CONTAINER;;
    esac

    if ! container_is_running "$container"; then
        err "'$container' container does not seem to be running. Trying anyway..."
    fi

    docker exec -it "$container" /bin/bash
}

#
# Import images from specified directory and launch containers from them.
# Current recreates the DB image.
#
import_containers() {
    dir="$1"
    [ -e "$dir" ] || mkdir -p "$dir"
    cd "$dir" || fatal "Could not cd to '$dir'"

    zcat "$UI_ARCHIVE" | docker load
    zcat "$DBA_ARCHIVE" | docker load

    run_ui_container
    run_db_container
    run_dba_container

    #zcat "$DB_ARCHIVE" | docker exec -i "$DB_CONTAINER" psql -U postgres

    cd -
}

#
# Export our containers, UI and DBA. DB
#
export_containers() {
    dir="$1"
    [ -e "$dir" ] || mkdir -p "$dir"
    cd "$dir" || fatal "Could not cd to '$dir'"

    docker save "$UI_CONTAINER" | gzip >"$UI_ARCHIVE"
    docker save "$DBA_CONTAINER" | gzip >"$DBA_ARCHIVE"

    #docker exec -t "$DB_CONTAINER" pg_dumpall -c -U postgres | gzip >"$DB_ARCHIVE"

    # FIXME: these exist for debugging efforts and can be removed later.
    chown ec2-user:ec2-user "$UI_ARCHIVE"
    chown ec2-user:ec2-user "$DB_ARCHIVE"
    chown ec2-user:ec2-user "$DBA_ARCHIVE"

    cd -
}

###############################################################################

main() {
    getargs "$@"
    get_input
    setup

    case "$MODE" in
        "fresh")
            clear_out_docker
            build_containers
            run_containers
            ;;
        "start")
            if [ -z "$STARTCONT" ]; then
                start_containers
            else
                start_container "$STARTCONT"
            fi
            ;;
        "stop")
            if [ -z "$STOPCONT" ]; then
                stop_container
            else
                stop_container "$STOPCONT"
            fi
            ;;
        "delete")
            if [ -z "$DELETECONT" ]; then
                clear_out_docker
            else
                clear_entities "$DELETECONT"
            fi
            ;;
        "connect")
            connect_container "$CONNECTTO"
            ;;
        "import")
            import_containers "$IMPORTDIR"
            ;;
        "export")
            export_containers "$EXPORTDIR"
            ;;
        "build")
            build_container "$BUILDCONT"
            ;;
        "run")
            run_container "$RUNCONT"
            ;;
    esac
}

###############################################################################

if [[ $(whoami) == "root" ]]; then
    main "$@"
else
    # We need root user for docker installation & commands
    sudo $0 "$@"
fi
