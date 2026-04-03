# Changelog

All notable changes to clawrelay-api will be documented in this file.

## [0.1.0.0] - 2026-04-03

### Added
- Multi-provider model support: MiniMax, Kimi (Moonshot), GLM (Zhipu) alongside Claude models
- `--model` flag to set default model at startup
- `--effort` passthrough to Claude CLI for controlling reasoning effort
- `--version` flag to display build version
- Version info in `/health` endpoint response
- Session viewer: inline event data in HTML for instant content rendering (fixes blank-page-on-load)
- Session viewer: bfcache reconnection support for back/forward navigation

### Changed
- `--proxy` flag replaces environment variable configuration for HTTP proxy
- WebSocket now only delivers live events; history rendered from inline data
- Session viewer status text updated from "auto-loading" to "waiting for new messages"

### Removed
- Redundant model aliases (sonnet, opus, haiku shorthand IDs)
