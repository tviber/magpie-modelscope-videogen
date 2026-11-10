# ModelScope WAN tools for Magpie

A standalone Magpie tools plugin and local Python service for submitting image-to-video jobs to WAN models through already signed-in ModelScope browser sessions.

The package runs locally. Magpie calls four tools; the Python service queues the request and continues polling ModelScope after the tool call returns:

- `wan_generate_flf2v`: generate from a first frame and a required final frame.
- `wan_generate_i2v`: generate from a first frame.
- `wan_get_video_status`: check progress.
- `wan_get_video_result`: download the completed video with authentication and return a usable local file path.

The service exposes `GET /v1/models`, `POST /v1/videos`, `GET /v1/videos/{id}` and `GET /v1/videos/{id}/content`. It binds to loopback by default. It does not require or store a ModelScope password or API key; it uses the browser sessions you configure.

## Requirements

- Magpie with its plugin tools enabled.
- Node.js 20 or later and npm for installing the local plugin dependencies.
- Python 3.11 or later, `pip` (or `uv`), and `ffprobe` from FFmpeg.
- Chrome or Chromium browser sessions signed in to ModelScope, each exposed on its own local CDP port.

## Install the Python service

From this package directory, create and activate a virtual environment.

macOS/Linux:

```sh
python3 -m venv .venv
. .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install the service in either environment:

```sh
python -m pip install --upgrade pip
python -m pip install -e .
```

Install FFmpeg if `ffprobe` is not already available. The service checks downloaded MP4 files with `ffprobe` before marking a job complete.

## Configure one or many ModelScope accounts

Start a separate Chrome/Chromium process with a separate user-data directory for each ModelScope account. Sign in to each profile and expose a different local CDP port. Keep each browser open while the service is running; do not share a profile between accounts. For example, on macOS:

```sh
open -na "Google Chrome" --args --remote-debugging-port=9223 --user-data-dir="$HOME/.modelscope-profiles/account-1"
```

For another account, use a different port and profile directory. Configure stable, non-personal account IDs mapped to CDP endpoints in `WAN_MODELSCOPE_ACCOUNTS`:

```sh
WAN_MODELSCOPE_ACCOUNTS='{"account-1":"http://127.0.0.1:9223"}' \
WAN_MEDIA_ROOTS="$HOME/wan-inputs" \
magpie-wan-service
```

Three accounts, for example:

```sh
WAN_MODELSCOPE_ACCOUNTS='{"account-1":"http://127.0.0.1:9223","account-2":"http://127.0.0.1:9224","account-3":"http://127.0.0.1:9225"}' \
WAN_MEDIA_ROOTS="$HOME/wan-inputs" \
magpie-wan-service
```

Windows PowerShell (after starting the signed-in browser profiles):

```powershell
$env:WAN_MODELSCOPE_ACCOUNTS = '{"account-1":"http://127.0.0.1:9223","account-2":"http://127.0.0.1:9224"}'
$env:WAN_MEDIA_ROOTS = "$HOME\wan-inputs"
$env:WAN_SERVICE_DATA_DIR = "$env:LOCALAPPDATA\magpie-wan"
magpie-wan-service
```

There is no two-account limit. The service starts one serial worker queue per configured account ID and assigns new jobs to the least-loaded queue. Open `https://modelscope.cn` or `https://www.modelscope.cn` and sign in in each browser profile. Other domains, HTTP pages, and redirects are rejected. Upload destinations must be official HTTPS ModelScope hosts or bucket-specific Alibaba OSS HTTPS endpoints; upload redirects are rejected.

Each ID is bound to a salted local fingerprint of its ModelScope login. The service checks the fingerprint before processing jobs, and the adapter checks the login before API operations. You can reorder the mapping or change a CDP port while keeping each ID with its original login. A different login or a duplicate identity blocks that worker. Use a new ID for a new account; never rename an ID with unfinished jobs. Choose generic IDs, since they appear in tool responses. Raw login identities and CDP addresses are not persisted.

The legacy comma-separated `WAN_MODELSCOPE_CDP_URLS` is still accepted when `WAN_MODELSCOPE_ACCOUNTS` is absent; it assigns `account-1`, `account-2`, etc. by position. Prefer the explicit mapping. Identity checks prevent a reordered legacy list from silently switching previously bound accounts. `WAN_MEDIA_ROOTS` is a path-separated list (`:` on macOS/Linux, `;` on Windows) of directories the service is allowed to read input frames from. It is required; use the narrowest directory that contains your frames. Frame paths passed to the tools must be absolute and inside one of these roots.

The service listens on `127.0.0.1:8765`. You can set `WAN_SERVICE_HOST`, `WAN_SERVICE_PORT`, `WAN_SERVICE_TOKEN`, `WAN_SERVICE_POLL_SECONDS`, and `WAN_SERVICE_DATA_DIR`. Keep the host on loopback: frame inputs are local paths. If you set `WAN_SERVICE_TOKEN`, also set the same token in the Magpie plugin options. Do not put tokens in this package or share the local service endpoint over a network.

## Install the Magpie plugin

From this package directory:

```sh
npm ci --ignore-scripts
magpie plugin add .
```

The folder install loads the source in place, so install its npm dependencies first. `@opencode-ai/plugin` is declared and locked; a pre-existing Bun cache is not an installation requirement. For a package extracted from `npm pack` (without the repository lockfile), use `npm install --ignore-scripts` instead of `npm ci`.

The default plugin service URL is `http://127.0.0.1:8765`. If you use a different local port, set its URL:

```sh
magpie plugin options magpie-modelscope-wan '{"serverUrl":"http://127.0.0.1:8766"}'
```

If you also set `WAN_SERVICE_TOKEN` for the service, give Magpie the same token in its local plugin options. Keep the real token out of shell history and source control.

As described in the [official plugin documentation](https://usemagpie.ai/docs/plugins), Magpie automatically adds a tools plugin as its own MCP server in Library → MCP. Assign it to the intended agents there, or use the MCP router. Plugins → Installed also offers Hide from agents. Manual MCP setup is needed only for clients outside this integration; the combined endpoint is `/mcp/magpie/tools` on your Magpie gateway.

These remain four dedicated tools, including first/last-frame generation; they do not register a native `generate_video` model. Each call must fit Magpie's 55-second limit, so generation stays in the Python worker. The plugin does not install or supervise that service; optional plugin pages and `page.service` do not replace service setup.

## Requests and output

Requests have no built-in style or negative-prompt defaults. The prompt is sent as provided; `style` and `negative_prompt` apply only when supplied on that request. `size` is optional and must preserve the frame aspect ratio; dimensions must be multiples of 16 and at most 1280 pixels. Without `size`, the service selects a supported size from the first frame. Duration is 1–10 whole seconds and defaults to 5.

The service copies inputs and completed videos to its private local data directory: `~/Library/Application Support/magpie-wan` on macOS, `$XDG_DATA_HOME/magpie-wan` or `~/.local/share/magpie-wan` on Linux, and `%LOCALAPPDATA%\magpie-wan` on Windows. Service status responses use relative `output_path` and `content_url` values. When authentication is enabled, that HTTP content URL requires a Bearer header.

Call `wan_get_video_result` to retrieve the file: the plugin supplies the token in a header, streams the video to `~/.magpie-wan/results/wan-*/video.mp4`, and returns an absolute `output_path` plus `file_url`. Set the plugin's `outputDir` option to change the local destination. This path is on the plugin host machine and can be opened by local agents; it is not a public download link. No token is included in a URL or tool response. Aborts, failed downloads and the 50-second retrieval timeout remove partial files; call the result tool again to retry. Completed copies remain until you delete them. First-to-last-frame input currently supports `wan2.1-flf2v-14b`.

Jobs persist across restarts. A service interruption during submission is marked `unknown_submission`; the service will not silently submit the job again. Inspect the matching ModelScope account, then reconcile the task:

```sh
curl -X POST http://127.0.0.1:8765/v1/videos/JOB_ID/resolve \
  -H 'Content-Type: application/json' \
  -d '{"provider_task_id":"MODELSCOPE_TASK_ID"}'
```

If you confirmed ModelScope did not create a task, explicitly requeue it:

```sh
curl -X POST http://127.0.0.1:8765/v1/videos/JOB_ID/resolve \
  -H 'Content-Type: application/json' \
  -d '{"confirm_no_task":true}'
```

When `WAN_SERVICE_TOKEN` is set, pass the matching bearer token to these requests too.

## Upgrade and migrate existing jobs

1. Stop the old service and back up the entire private data directory. Copy it only while the service is stopped, including the database, any SQLite sidecars, input frames and videos.
2. On the original machine, configure `WAN_MODELSCOPE_ACCOUNTS` with the existing `account-1`, `account-2`, etc. IDs and their original browser logins. Old databases do not contain identity evidence, so the first association must be checked manually. After checking, run `magpie-wan-service --bind-accounts` once, then back up/copy the stopped service data again. This command binds previously unbound identities; it cannot override a mismatch.
3. On the new machine, install the Python service, ffprobe and plugin dependencies. Set `WAN_SERVICE_DATA_DIR` to the copied directory, configure new media roots, and map the same account IDs to the new browser endpoints. Sign in to the same ModelScope accounts. Login fingerprints and their salt travel in the database; browser cookies do not.
4. Start the service. Existing Windows backslash paths are converted to forward slashes automatically; relative paths resolve on Windows, macOS and Linux. Queued inputs already copied to the data directory do not require the original source paths. A missing account ID or changed identity cannot silently resume another account's jobs.

If the old machine is unavailable, perform step 2 on the destination only after manually confirming the original account-to-job correspondence. Without that evidence, leave those jobs blocked. Newly created databases bind accounts automatically on their first worker connection.

Expired video URLs (401/403, or three other failed downloads/validations) are cleared. The worker queries the original provider task for a fresh URL and continues downloading without generating again. `unknown_submission` still requires the explicit reconciliation described above.

## Validation and release

Run `python -m pip install -e '.[test]'`, `python -m pytest -q`, and `npm test` for the offline regression suite. These tests use mocked provider sessions and local HTTP servers, not browsers. The CI workflow checks Linux and Windows, installs an npm tarball into a clean directory with an isolated npm cache, imports its four tools, and installs the Python service from the same packaged source. A configured workflow is not evidence of a successful run: require it to pass before release. Real ModelScope login, upload and generation still need a separate end-to-end check on a suitable machine.

Before distributing, run `npm pack --dry-run --json` and inspect the whitelist (entry point, package metadata, Python source and documentation only). `npm pack` creates the distributable tarball; a recipient can extract it, follow the Python setup above, install npm dependencies, and run `magpie plugin add .`. No npm registry release or public repository is assumed to exist. Publishing requires a chosen license, release destination and a successful clean-install CI run; this change does not publish anything. Never distribute the runtime data directory, browser profiles or an unrelated workspace.

## Privacy

See [PRIVACY.md](PRIVACY.md) for data flow, local storage, and the privacy audit scope.
