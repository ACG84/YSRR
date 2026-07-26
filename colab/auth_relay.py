#!/usr/bin/env python3
"""Authorise `colab` from a headless container, in two steps.

`colab-cli` authenticates with a copy-paste OAuth flow -- not a localhost
redirect, and not the OOB flow Google blocked in 2022. It prints a URL, you
approve in any browser, Google shows an authorization code on its own landing
page, and you paste it back. That last part normally reads from a TTY, which is
what stops this container from just running `colab new`.

Nothing about the flow itself requires a TTY, so this splits it in half:

    python colab/auth_relay.py url                 # -> paste this URL in a browser
    python colab/auth_relay.py exchange <CODE>     # -> writes ~/.config/colab-cli/token.json

The flow uses PKCE with an auto-generated verifier, so the two halves have to
share state. `url` stashes the verifier and the CSRF `state` in
``~/.config/colab-cli/relay_state.json``; `exchange` reads them back. That file
is a live half-finished credential -- it is deleted on success, and should be
deleted by hand if you abandon the flow.

Scopes requested are exactly the CLI's own (``colab_cli.auth.PUBLIC_SCOPES``),
which include ``cloud-platform`` and ``drive.file``. That is broad. The
resulting token lands in an ephemeral container that is reclaimed on idle; to
revoke early, use https://myaccount.google.com/permissions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Run under the CLI's own interpreter so the client config and scopes cannot
# drift from what `colab` itself expects.
CLI_ROOT = Path("/root/.local/share/uv/tools/google-colab-cli")
SITE_PACKAGES = CLI_ROOT / "lib/python3.12/site-packages"
if SITE_PACKAGES.is_dir() and str(SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(SITE_PACKAGES))

from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: E402

from colab_cli.auth import (  # noqa: E402
    PUBLIC_SCOPES,
    REMOTE_REDIRECT_URI,
    TOKEN_CONFIG_PATH,
)

RELAY_STATE = Path(os.path.expanduser("~/.config/colab-cli/relay_state.json"))


def client_config() -> dict:
    from importlib import resources

    return json.loads(
        resources.files("colab_cli").joinpath("oauth_config.json").read_text()
    )


def make_flow(code_verifier: str | None = None) -> InstalledAppFlow:
    flow = InstalledAppFlow.from_client_config(
        client_config(),
        PUBLIC_SCOPES,
        code_verifier=code_verifier,
        autogenerate_code_verifier=code_verifier is None,
    )
    flow.redirect_uri = REMOTE_REDIRECT_URI
    return flow


def cmd_url() -> int:
    flow = make_flow()
    auth_url, state = flow.authorization_url(prompt="consent", token_usage="remote")

    RELAY_STATE.parent.mkdir(parents=True, exist_ok=True)
    RELAY_STATE.write_text(
        json.dumps({"code_verifier": flow.code_verifier, "state": state})
    )
    RELAY_STATE.chmod(0o600)

    print("Open this in any browser, approve, and copy the code Google shows:\n")
    print(auth_url)
    print("\nThen run:  python colab/auth_relay.py exchange <CODE>")
    return 0


def cmd_exchange(code: str) -> int:
    if not RELAY_STATE.exists():
        print(f"no pending flow at {RELAY_STATE} -- run `url` first", file=sys.stderr)
        return 1
    saved = json.loads(RELAY_STATE.read_text())

    flow = make_flow(code_verifier=saved["code_verifier"])
    flow.fetch_token(code=code.strip())

    token_path = Path(TOKEN_CONFIG_PATH)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(flow.credentials.to_json())
    token_path.chmod(0o600)
    RELAY_STATE.unlink()

    print(f"token written to {token_path}")
    print("`colab` will now pick it up; verify with:  colab ls")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("url", help="print the authorization URL")
    ex = sub.add_parser("exchange", help="redeem the authorization code")
    ex.add_argument("code")
    args = p.parse_args()

    return cmd_url() if args.cmd == "url" else cmd_exchange(args.code)


if __name__ == "__main__":
    raise SystemExit(main())
