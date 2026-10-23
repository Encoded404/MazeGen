# Infinite Runner secrets

Files in this folder are sealed at build time by `secrets-gen` into the
generated `Examples.InfiniteRunner.Secrets` module and are **not** committed.
Only this README, `manifest.txt` and `*.example` files are tracked.

## Layout

| File | Committed | Purpose |
|---|---|---|
| `.keyring` | no | key id -> 32-byte hex key, used to seal at build time |
| `manifest.txt` | yes | optional per-file variant / key id / AAD override |
| `leaderboard_endpoint.txt` | no | `host:port` the client connects to |
| `leaderboard_server_pubkey.txt` | no | pinned server X25519 public key (hex, 32 bytes) |
| `*.example` | yes | templates describing the real files' formats |

## `.keyring` format

One key per line: `<key_id> <64 hex chars>`. Lines starting with `#` are
ignored. Multiple ids may be live at once; the envelope records which key was
used, so rotating means adding a new id and re-sealing new files.

```
# id  key
1    000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
```

Copy `.keyring.example` to `.keyring` and replace the key with a random 32-byte
value (for example `openssl rand -hex 32`).

## `manifest.txt` format

```
# <file> <cipher_variant> <key_id> <aad>
leaderboard_endpoint.txt 1 1 endpoint
```

`cipher_variant` is `1` for XChaCha20-Poly1305 (default) or `2` for
ChaCha20-Poly1305 (IETF). The AAD defaults to the file name; set it explicitly
to bind a blob to a purpose. Unlisted files use the defaults.

## Server identity and key pinning

The leaderboard server has a long-term X25519 keypair. Its private half lives
only on the server; the client pins the public half, which is what authenticates
the server during the handshake.

For local development, put the server's private identity in this folder as
`.server_identity` (hex, 32 bytes). At build time `secrets-gen` derives
`leaderboard_server_pubkey.txt` from it automatically, so the pinned key can
never drift from the server you run.

```
# one-time: create the identity (the server prints its public key)
./build/bin/server/Debug/leaderboard_server --port 7777 \
    --identity examples/infinite_runner/secrets/.server_identity

# from then on, always start the server with that identity:
./build/bin/server/Debug/leaderboard_server --port 7777 \
    --identity examples/infinite_runner/secrets/.server_identity
# or set VKENGINE_SERVER_IDENTITY once instead of passing --identity
```

`--identity <file>` overrides where the private half is kept (default
`leaderboard_server.identity` beside the working directory, or
`<store>.identity` when `--store` is given). If the server starts with a
different identity than the one the client pinned, every handshake is refused
and the client reports `server identity mismatch: pinned <a>..., server offered
<b>...`. Set the log level to `warn` or lower to see it (see below).

If `.server_identity` is absent, an explicit `leaderboard_server_pubkey.txt` is
used instead — that is the path for pinning a remote server whose private key
you do not hold. Because either source is sealed into the client at build time,
**changing the identity requires a rebuild** of the game. If neither is present
the client cannot authenticate and stays offline; play on a local profile
instead.

The server also needs the ruleset fingerprint it is serving to be listed in
`leaderboard/accepted_configs.txt` (see that file). Keeping a previous
fingerprint there lets clients built against the previous balance keep playing.

## Logging

The leaderboard logs to stderr on the server, and to the engine logger in the
game. Levels are `trace`, `debug`, `info` (default), `warn`, `error` and `off`:

```
leaderboard_server --log-level debug
VKENGINE_LEADERBOARD_LOG=debug ./build/bin/examples/Debug/infinite_runner/infinite_runner
```

`info` reports connections, handshakes, registrations, logins and accepted
scores; `debug` adds per-request traffic and reconnect detail; `warn` reports
unreachable servers, pin mismatches, rejected credentials and dropped frames.

## Security note

The sealing key is embedded in the produced binary so the client can open the
blobs. This is obfuscation with tamper-evidence, not secrecy: it keeps secrets
out of git and out of `strings`, and it makes edited blobs fail loudly. It does
not protect against a determined reverse engineer. The registration token, by
contrast, is a real per-account credential; it is stored owner-only on disk and
only ever sent inside the X25519-protected channel.
