#!/usr/bin/env bash
# Bash completion for Rhiza make targets
#
# Installation:
#   Source this file in your ~/.bashrc or ~/.bash_profile:
#     source /path/to/.rhiza/completions/rhiza-completion.bash
#
#   Or copy to bash completion directory:
#     sudo cp .rhiza/completions/rhiza-completion.bash /etc/bash_completion.d/rhiza
#

# Return 0 (stale) when the cache file is missing or any makefile source
# changed since it was written.
_rhiza_make_cache_stale() {
    local cache_file="$1" src
    [[ -f "$cache_file" ]] || return 0
    for src in Makefile local.mk .rhiza/rhiza.mk .rhiza/make.d/*.mk; do
        [[ -f "$src" && "$src" -nt "$cache_file" ]] && return 0
    done
    return 1
}

_rhiza_make_completion() {
    local cur prev opts cache_dir cache_file
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Check if we're in a directory with a Makefile
    if [[ ! -f "Makefile" ]]; then
        return 0
    fi

    # Target extraction parses the full make database (make -qp), which is
    # slow on large Makefiles - cache the result per directory and refresh
    # only when a makefile source changes.
    cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/rhiza"
    cache_file="$cache_dir/targets-$(pwd | cksum | cut -d' ' -f1)"

    if _rhiza_make_cache_stale "$cache_file" && mkdir -p "$cache_dir" 2>/dev/null; then
        # Extract make targets from Makefile and all included .mk files
        make -qp 2>/dev/null | \
            awk -F':' '/^[a-zA-Z0-9][^$#\/\t=]*:([^=]|$)/ {split($1,A,/ /);for(i in A)print A[i]}' | \
            grep -v '^Makefile$' | \
            sort -u > "$cache_file"
    fi

    if [[ -r "$cache_file" ]]; then
        opts=$(cat "$cache_file")
    else
        # Cache unavailable (e.g. unwritable HOME): fall back to direct parsing
        opts=$(make -qp 2>/dev/null | \
               awk -F':' '/^[a-zA-Z0-9][^$#\/\t=]*:([^=]|$)/ {split($1,A,/ /);for(i in A)print A[i]}' | \
               grep -v '^Makefile$' | \
               sort -u)
    fi

    # Add common make variables that can be overridden
    local vars="DRY_RUN=1 ENV=dev ENV=staging ENV=prod"
    opts="$opts $vars"

    # Generate completions
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

# Register the completion function for make command
complete -F _rhiza_make_completion make

# Also complete for direct make invocation with path
complete -F _rhiza_make_completion ./Makefile

# Helpful aliases (optional - uncomment if desired)
# alias m='make'
# complete -F _rhiza_make_completion m
