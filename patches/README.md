# Magpie wallet trend chart

`magpie-0.1.3912-wallet-trends.patch` targets the official Magpie
`v0.1.3912` source. This is a native Magpie change: the plugin's standard
usage interface cannot add native Usage-page charts on its own.

The patch records actual prepaid-wallet readings, exposes the history on
`windows[].balanceTrend`, and reuses Magpie's balance chart. Expanded
accounts show their own histories; the folded provider shows their sum.
Only matching wallet names and units can be combined, starting once all
accounts have a reading. Missing or stale accounts are never treated as
zero. No quota ceiling, renewal, or past history is fabricated. A first
reading is a point. The duplicate top-level balance row remains absent.

History stays in Magpie's existing `balance-history.json`, with the same
45-day retention, 14-day display and sampling rules as key balances.
Wallet identities use hashes under a separate namespace. Adding/removing
accounts changes the membership of the total: totals are recalculated for
the accounts currently configured. The combined curve has no forecast.

Apply and build from a clean checkout of the matching official source:

```sh
git apply --check /absolute/path/to/magpie-0.1.3912-wallet-trends.patch
git apply /absolute/path/to/magpie-0.1.3912-wallet-trends.patch
make build-release VERSION=0.1.3912
```

Use the source's supported Go toolchain (1.26.8) from a standalone SDK.
A toolchain installed inside GOMODCACHE cannot use the upstream Darwin
runtime overlay; copy it outside that cache or use the official standalone
Go SDK. On macOS, back up the original application, replace the binary in
a staged copy, sign that copy, verify it, then install and restart normally.
The local build is ad-hoc signed and is not Apple's notarized release.
An official Magpie update can replace it; rebase and retest this patch
against that version before applying again.

Verification: provider package tests; targeted wallet/balance tests with
`-race`; GUI core budget; wallet, existing balance-curve and wallet-display
browser tests in Chromium and WebKit at desktop/mobile widths and five
languages. Regression proof: removing `noteWalletHistory` from
`noteCardHistory` makes `TestWalletHistoryAccountIsolationAndReport` fail
because account 0 loses its wallet history.

Local verification limits: the full native Go suite did not complete (the
gateway test package stalled). The provider package passed in full. The
GUI package has two failures reproduced on the unmodified v0.1.3912
source: `TestLaneLimitRoutes` sees leftover busy lanes and
`TestProxyCheckSaysAProxyPassedBy` receives HTTP 502 in this environment.
Focused wallet/history/report/core-budget tests, race checks and vet passed.
The broad `usage-all-accounts.test.cjs` browser run stalled and was stopped;
the 12 focused wallet/balance browser cases passed. No claim of a fully
green native suite is made. Local app verified after restart with four real
wallet histories and the folded 1,040 魔粒 total; WAN service remained healthy.

## Plugin video submissions in Routing

`magpie-0.1.3912-plugin-video-routing.patch` is independent of the wallet
patch and can be applied after it. It adds optional `magpie.route` metadata
to tool definitions. Only the WAN generation tools opt in; status and
result tools remain ordinary tools. It works through the all-plugin MCP
endpoint, per-plugin endpoint and Library MCP hub. A route records the
calling agent, explicit session and gateway key, selected model/account,
submission duration and outcome. Its responseId retains the WAN job ID.
Recent Request and response details contain the job's returned state and
an explicit generation_completed flag. An accepted queued job is a
successful submission, not a finished video. This does not backfill old
jobs or monitor the generation lifecycle in Routing. No generation is
repeated for logging, and no token, dollar or magicube charge is inferred.

Checks: native plugin host metadata, all MCP paths, no duplicate poll
records, caller-key attribution, cancellation and failure tests; race
checks; vet; 8 Routing browser cases in Chromium/WebKit, English/Chinese,
1100px/420px, opening request/response details without runtime errors.
Removing the direct MCP routing hook makes the new regression fail with
`video submission missing from Routing: []`.

Applied locally to Magpie 0.1.3912 alongside the wallet patch. A real MCP
call using an intentionally missing input frame failed before submission
and appeared in the native Routing list as Codex / WAN / 502 with its
duration. This check generated no video and incurred no provider charge.
The WAN service remained healthy after restart. Both patches apply cleanly
to the official v0.1.3912 source; the original local application is backed
up before replacement.
