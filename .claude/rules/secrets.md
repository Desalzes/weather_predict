# Secret Handling Rules

## Never Commit These Files

- `config.json` — local config with API keys (gitignored)
- `config.local.json` — alternate local override
- `api-credentials.txt` — Kalshi RSA private key (gitignored)
- `*.pem`, `*.key` — any private key material

## Environment Variables Carrying Secrets

- `DEEPSEEK_API_KEY` — DeepSeek chat-completions API
- `KALSHI_API_KEY_ID` — can be set via env instead of config
- `NCEI_API_TOKEN` — NOAA Climate Data Online access

## Config Inheritance

`config.example.json` is committed and contains safe defaults (no real keys).
`config.json` is gitignored and overrides example defaults at runtime.
Missing top-level keys in `config.json` inherit from the example automatically.

## Safe Template

`config.example.json` is the committed template. It must never contain real
API keys, tokens, or private key paths that resolve to actual files. Placeholder
values like `"YOUR_KEY_HERE"` or empty strings are acceptable.

## When Reorganizing

If the repo structure changes, preserve all local credential files and config
material. Credentials are more important than folder layout.
