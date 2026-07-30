#compdef make
# Zsh completion for Rhiza make targets
#
# Installation:
#   Add this file to your fpath and ensure compinit is called:
#
#   Method 1 (User-local):
#     mkdir -p ~/.zsh/completion
#     cp .rhiza/completions/rhiza-completion.zsh ~/.zsh/completion/_make
#     Add to ~/.zshrc:
#       fpath=(~/.zsh/completion $fpath)
#       autoload -U compinit && compinit
#
#   Method 2 (Source directly):
#     Add to ~/.zshrc:
#       source /path/to/.rhiza/completions/rhiza-completion.zsh
#
#   Method 3 (System-wide):
#     sudo cp .rhiza/completions/rhiza-completion.zsh /usr/local/share/zsh/site-functions/_make
#

# Return 0 (stale) when the cache file is missing or any makefile source
# changed since it was written.
_rhiza_make_cache_stale() {
    local cache_file="$1" src
    [[ -f "$cache_file" ]] || return 0
    for src in Makefile local.mk .rhiza/rhiza.mk .rhiza/make.d/*.mk(N); do
        [[ -f "$src" && "$src" -nt "$cache_file" ]] && return 0
    done
    return 1
}

_rhiza_make() {
    local -a targets variables
    local cache_dir cache_file

    # Check if we're in a directory with a Makefile
    if [[ ! -f "Makefile" ]]; then
        return 0
    fi

    # Target extraction parses the full make database (make -qp) twice, which
    # is slow on large Makefiles - cache both lists per directory and refresh
    # only when a makefile source changes.
    cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/rhiza"
    cache_file="$cache_dir/targets-$(pwd | cksum | cut -d' ' -f1)"

    if _rhiza_make_cache_stale "$cache_file.desc" && mkdir -p "$cache_dir" 2>/dev/null; then
        # Extract make targets with descriptions (format: target:description)
        make -qp 2>/dev/null | \
        awk -F':' '
            /^# Files/,/^# Finished Make data base/ {
                if (/^[a-zA-Z0-9_-]+:.*##/) {
                    target=$1
                    desc=$0
                    sub(/^[^#]*## */, "", desc)
                    gsub(/^[ \t]+/, "", target)
                    print target ":" desc
                }
            }
        ' | \
        grep -v '^Makefile:' | \
        sort -u > "$cache_file.desc"

        # Also get targets without descriptions
        make -qp 2>/dev/null | \
        awk -F':' '/^[a-zA-Z0-9_-]+:([^=]|$)/ {
            split($1,A,/ /)
            for(i in A) print A[i]
        }' | \
        grep -v '^Makefile$' | \
        sort -u > "$cache_file.plain"
    fi

    local -a plain_targets
    if [[ -r "$cache_file.desc" ]]; then
        targets=(${(f)"$(cat "$cache_file.desc")"})
        plain_targets=(${(f)"$(cat "$cache_file.plain" 2>/dev/null)"})
    else
        # Cache unavailable (e.g. unwritable HOME): fall back to direct parsing
        plain_targets=(${(f)"$(
            make -qp 2>/dev/null | \
            awk -F':' '/^[a-zA-Z0-9_-]+:([^=]|$)/ {
                split($1,A,/ /)
                for(i in A) print A[i]
            }' | \
            grep -v '^Makefile$' | \
            sort -u
        )"})
    fi

    # Common make variables
    variables=(
        'DRY_RUN=1:preview mode without making changes'
        'ENV=dev:development environment'
        'ENV=staging:staging environment'
        'ENV=prod:production environment'
        'COVERAGE_FAIL_UNDER=:minimum coverage threshold'
        'PYTHON_VERSION=:override Python version'
    )

    # Combine all completions
    local -a all_completions
    all_completions=($targets $plain_targets $variables)

    # Show completions with descriptions
    _describe 'make targets' all_completions
}

# Register the completion function
compdef _rhiza_make make

# Optional: Add completion for common aliases
# Uncomment these if you use these aliases
# alias m='make'
# compdef _rhiza_make m
