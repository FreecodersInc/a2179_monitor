#!/usr/bin/env bash
# Expand only the current user's literal ~ prefix. Never eval configuration values.
expand_home_path() {
  case "$1" in
    '~') printf '%s' "$HOME" ;;
    '~/'*) printf '%s/%s' "$HOME" "${1:2}" ;;
    *) printf '%s' "$1" ;;
  esac
}
