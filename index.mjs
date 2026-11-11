import { tool } from "@opencode-ai/plugin"
import { constants, createWriteStream } from "node:fs"
import { createHash } from "node:crypto"
import { chmod, copyFile, mkdir, mkdtemp, readFile, realpath, rename, rm, stat, writeFile } from "node:fs/promises"
import { homedir } from "node:os"
import { basename, isAbsolute, join, relative, sep } from "node:path"
import { Readable } from "node:stream"
import { pipeline } from "node:stream/promises"
import { fileURLToPath, pathToFileURL } from "node:url"
import { execFile } from "node:child_process"
import { promisify } from "node:util"
import { setTimeout as delay } from "node:timers/promises"

const DEFAULT_SERVER_URL = "http://127.0.0.1:8765"

function serviceUrl(options) {
  const base = String(options?.serverUrl || DEFAULT_SERVER_URL).replace(/\/+$/, "")
  if (!/^https?:\/\//i.test(base)) throw new Error("WAN serviceUrl must use http:// or https://")
  return base
}

function headers(options) {
  const result = { "content-type": "application/json" }
  if (options?.token) result.authorization = `Bearer ${options.token}`
  return result
}

async function mediaRoots(options, signal) {
  return (await request(options, "/v1/media/roots", { signal })).roots || []
}

function insideRoot(path, roots) {
  return roots.some(root => {
    const rel = relative(root, path)
    return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel))
  })
}

async function prepareFrame(path, options, signal, copyToRoot, roots, created) {
  if (!isAbsolute(path)) throw new Error("WAN frame paths must be absolute local paths")
  // Resolve symlinks and '..' before checking containment; the service checks again.
  const source = await realpath(path)
  const info = await stat(source)
  if (!info.isFile() || info.size > 30 * 1024 * 1024) throw new Error("WAN frames must be regular files no larger than 30 MiB")
  const allowed = await Promise.all(roots.map(root => realpath(root)))
  let candidate = source
  if (!insideRoot(source, allowed)) {
    if (!copyToRoot) {
      throw new Error(`Frame is outside WAN_MEDIA_ROOTS. Allowed directories: ${roots.join(", ")}. Set copy_to_media_root=true to copy this image into the first allowed directory.`)
    }
    const host = new URL(serviceUrl(options)).hostname
    if (!["127.0.0.1", "localhost", "[::1]"].includes(host)) throw new Error("Automatic frame copying requires a service on this computer")
    if (!allowed.length) throw new Error("WAN service did not report an input directory")
    const directory = await mkdtemp(join(allowed[0], ".magpie-wan-"))
    created.push(directory)
    candidate = join(directory, basename(source))
    await copyFile(source, candidate, constants.COPYFILE_EXCL)
    await chmod(candidate, 0o600)
  }
  signal?.throwIfAborted()
  await request(options, "/v1/frames/validate", {
    method: "POST", body: JSON.stringify({ first_frame: candidate }), signal,
  })
  return candidate
}

async function prepareFrames(args, options, signal) {
  const roots = await mediaRoots(options, signal)
  const created = []
  try {
    const first_frame = await prepareFrame(args.first_frame, options, signal, args.copy_to_media_root === true, roots, created)
    const last_frame = args.last_frame
      ? await prepareFrame(args.last_frame, options, signal, args.copy_to_media_root === true, roots, created) : undefined
    return { first_frame, last_frame, copied_directories: created, allowed_roots: roots }
  } catch (error) {
    await Promise.all(created.map(directory => rm(directory, { recursive: true, force: true })))
    throw error
  }
}

async function request(options, path, init = {}) {
  const response = await fetch(`${serviceUrl(options)}${path}`, {
    ...init,
    headers: { ...headers(options), ...(init.headers || {}) },
    redirect: "error",
    signal: AbortSignal.any([init.signal, AbortSignal.timeout(10_000)].filter(Boolean)),
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = payload?.error?.message || `WAN service returned HTTP ${response.status}`
    throw Object.assign(new Error(message), { status: response.status, code: payload?.error?.code })
  }
  return payload
}

function generateArgs(includeLastFrame, defaultModel) {
  const fields = {
    prompt: tool.schema.string().min(1).describe("Action and visual content for this clip."),
    first_frame: tool.schema.string().min(1).describe("Absolute local path to the first frame image."),
    seconds: tool.schema.number().int().min(1).max(10).optional().describe("Duration in whole seconds (1–10). Defaults to 5."),
    size: tool.schema.string().optional().describe("Optional output dimensions such as 1280x720; must preserve the frame aspect ratio."),
    model: tool.schema.string().optional().describe(`Model id; defaults to ${defaultModel}.`),
    style: tool.schema.string().optional().describe("Optional style text supplied for this request only."),
    negative_prompt: tool.schema.string().optional().describe("Optional negative prompt supplied for this request only."),
    seed: tool.schema.number().int().min(-1).optional().describe("Seed, or -1 for the provider default."),
    project: tool.schema.string().max(160).optional().describe("Optional project or shot group name."),
    shot: tool.schema.string().max(160).optional().describe("Optional shot name within the project."),
    account: tool.schema.string().optional().describe("Optional configured ModelScope account ID, such as account-1."),
    copy_to_media_root: tool.schema.boolean().optional().describe("Copy frames into the service's allowed input directory when they are outside it."),
    idempotency_key: tool.schema.string().optional().describe("Optional key to return the same job for retries of this exact request."),
  }
  if (includeLastFrame) {
    fields.last_frame = tool.schema.string().min(1).describe("Absolute local path to the last frame image.")
  }
  return fields
}

function submitTool(options, includeLastFrame, defaultModel, watch) {
  return tool({
    description: includeLastFrame
      ? "Submit a WAN first-frame-to-last-frame video job. Returns a durable job id immediately; use wan_get_video_status to check progress."
      : "Submit a WAN first-frame-to-video job. Returns a durable job id immediately; use wan_get_video_status to check progress.",
    args: generateArgs(includeLastFrame, defaultModel),
    async execute(args, ctx = {}) {
      const body = {
        prompt: args.prompt,
        first_frame: args.first_frame,
        last_frame: includeLastFrame ? args.last_frame : undefined,
        seconds: args.seconds,
        size: args.size,
        model: args.model || defaultModel,
        style: args.style,
        negative_prompt: args.negative_prompt,
        seed: args.seed,
        project: args.project,
        shot: args.shot,
        account: args.account,
        idempotency_key: args.idempotency_key,
      }
      const prepared = await prepareFrames({ ...args, last_frame: includeLastFrame ? args.last_frame : undefined }, options, ctx.abort)
      body.first_frame = prepared.first_frame
      body.last_frame = prepared.last_frame
      // The service snapshots these inputs before it acknowledges the job. Keep
      // staging copies on an ambiguous network failure so an idempotent retry works.
      const result = await request(options, "/v1/videos", {
        method: "POST",
        headers: args.idempotency_key ? { "Idempotency-Key": args.idempotency_key } : {},
        body: JSON.stringify(body),
        signal: ctx.abort,
      })
      try { await watch?.(result.id) } catch {
        // Submission has already succeeded: never make a notification failure
        // look like a failed submission that should generate a second video.
        result.notification_warning = "自动完成提醒暂不可用；请用任务 ID 查询状态"
      }
      return { ...result, prepared_frames: prepared }
    },
  })
}

function statusTool(options) {
  return tool({
    description: "Check a WAN job without waiting for generation. Includes phase, progress, requested and actual video specifications when available.",
    args: { job_id: tool.schema.string().min(1).describe("Job id returned by a WAN submit tool.") },
    async execute({ job_id }, ctx = {}) {
      return await request(options, `/v1/videos/${encodeURIComponent(job_id)}`, { signal: ctx.abort })
    },
  })
}

function listTool(options) {
  return tool({
    description: "List WAN jobs with prompts, input frames and specifications; filter by project and paginate with next_offset.",
    args: {
      project: tool.schema.string().optional(),
      limit: tool.schema.number().int().min(1).max(200).optional(),
      offset: tool.schema.number().int().min(0).optional(),
    },
    async execute(args, ctx = {}) {
      const query = new URLSearchParams(Object.entries(args).filter(([, value]) => value !== undefined))
      return request(options, `/v1/videos?${query}`, { signal: ctx.abort })
    },
  })
}

const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]))
const hash = value => createHash("sha256").update(value).digest("hex").slice(0, 24)

function resultRoot(options) {
  const root = options?.outputDir || join(homedir(), ".magpie-wan", "results")
  if (!isAbsolute(root)) throw new Error("WAN outputDir must be an absolute local directory")
  return root
}

async function streamFile(options, route, output, signal) {
  const response = await fetch(`${serviceUrl(options)}${route}`, { headers: headers(options), signal, redirect: "error" })
  if (!response.ok || !response.body) {
    await response.body?.cancel()
    throw new Error(`WAN content download returned HTTP ${response.status}`)
  }
  await pipeline(Readable.fromWeb(response.body), createWriteStream(`${output}.partial`, { flags: "wx", mode: 0o600 }), { signal })
  signal.throwIfAborted()
  await rename(`${output}.partial`, output)
}

async function downloadResult(options, jobId, abort) {
  const signal = AbortSignal.any([abort, AbortSignal.timeout(50_000)].filter(Boolean))
  const route = `/v1/videos/${encodeURIComponent(jobId)}`
  const result = await request(options, route, { signal })
  if (result.status !== "completed") return result
  const root = resultRoot(options)
  await mkdir(root, { recursive: true, mode: 0o700 })
  const directory = await mkdtemp(join(root, "wan-"))
  const output = join(directory, "video.mp4")
  try {
    // Never forward authentication to URLs supplied in a response.
    await streamFile(options, `${route}/content`, output, signal)
    const frames = []
    for (const frame of result.input_frames || []) {
      if (!["first_frame", "last_frame"].includes(frame.role)) continue
      const local = join(directory, `${frame.role}.png`)
      await streamFile(options, `${route}/frames/${frame.role}`, local, signal)
      frames.push({ role: frame.role, name: frame.name, width: frame.width, height: frame.height,
        output_path: local, file_url: pathToFileURL(local).href })
    }
    const manifest = {
      id: result.id, status: result.status, project: result.project, shot: result.shot,
      account_id: result.account_id, model: result.model, prompt: result.prompt,
      style: result.style, negative_prompt: result.negative_prompt,
      input_frames: frames, requested: result.requested, actual: result.actual,
      spec_warnings: result.spec_warnings, estimate: result.estimate, estimate_unit: result.estimate_unit,
      estimate_kind: result.estimate_kind, output_path: output, file_url: pathToFileURL(output).href,
      manifest_path: join(directory, "manifest.json"), preview_path: join(directory, "preview.html"),
      preview_url: pathToFileURL(join(directory, "preview.html")).href,
    }
    const actual = result.actual
    const measured = actual ? `${actual.duration ?? "?"} 秒 · ${actual.width ?? "?"} × ${actual.height ?? "?"} · ${actual.fps ?? "?"} fps` : "实测规格暂不可用"
    const requested = result.requested
    const html = `<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src 'self' file:; img-src 'self' file:; style-src 'unsafe-inline'"><title>WAN · ${escapeHtml(result.shot || result.id)}</title>
<style>body{margin:32px auto;padding:0 24px;max-width:1000px;background:#141b20;color:#eff0e8;font:16px/1.6 Georgia,serif}h1{font-size:32px;margin-bottom:8px}video{display:block;width:100%;max-height:65vh;background:#000;border-radius:8px}p{white-space:pre-wrap;overflow-wrap:anywhere}.frames{display:flex;gap:16px;flex-wrap:wrap}.frames img{max-width:100%;max-height:180px}figure{margin:0;max-width:45%}small{color:#b7c9ca}a{color:#91d5c4}.warning{color:#f2cf8d}</style>
<small>WAN / ${escapeHtml(result.project || "未分组")} / ${escapeHtml(result.id)}</small><h1>${escapeHtml(result.shot || "视频已完成")}</h1><video controls preload="metadata" src="video.mp4"></video>
<p>请求：${escapeHtml(requested ? `${requested.seconds} 秒 · ${requested.size} · 帧率由模型决定` : "未知")}<br>实际：${escapeHtml(measured)}</p><p class="warning">${escapeHtml((result.spec_warnings || []).join("\n"))}</p>
<h2>提示词</h2><p>${escapeHtml(result.prompt)}</p><div class="frames">${frames.map(frame => `<figure><img src="${frame.role}.png" alt="${escapeHtml(frame.role)}"><figcaption>${escapeHtml(frame.name || frame.role)}</figcaption></figure>`).join("")}</div>
<p><a href="video.mp4">打开视频</a> · <a href="manifest.json">任务与素材清单</a></p></html>`
    await writeFile(manifest.manifest_path, JSON.stringify(manifest, null, 2), { mode: 0o600 })
    await writeFile(manifest.preview_path, html, { mode: 0o600 })
    return manifest
  } catch (error) {
    await rm(directory, { recursive: true, force: true })
    throw error
  }
}

function resultTool(options) {
  return tool({
    description: "Download the completed video, frame previews, manifest and HTML preview. Returns requested/actual specs and prompt; unfinished jobs return status.",
    args: { job_id: tool.schema.string().min(1) },
    execute: ({ job_id }, ctx = {}) => downloadResult(options, job_id, ctx.abort),
  })
}

// Persist watched IDs so a restarted plugin resumes notification delivery.
function completionWatcher(input, options) {
  if (!input?.client?.tui?.showToast || options.watchJobs === false) return undefined
  const stateFile = join(resultRoot(options), `watch-${hash(serviceUrl(options))}.json`)
  let pending = new Set(), timer, busy = false
  let writes = Promise.resolve()
  const persist = () => {
    const body = JSON.stringify([...pending])
    writes = writes.catch(() => {}).then(async () => {
      await mkdir(resultRoot(options), { recursive: true, mode: 0o700 })
      await writeFile(`${stateFile}.partial`, body, { mode: 0o600 })
      await rename(`${stateFile}.partial`, stateFile)
    })
    return writes
  }
  const schedule = () => {
    if (timer || !pending.size) return
    timer = setTimeout(tick, 15_000)
    timer.unref?.()
  }
  const ready = readFile(stateFile, "utf8").then(text => {
    const ids = JSON.parse(text)
    if (Array.isArray(ids)) pending = new Set(ids.filter(id => typeof id === "string"))
    schedule()
  }).catch(() => {})
  async function tick() {
    timer = undefined
    if (busy) return
    busy = true
    try {
      for (const id of [...pending]) {
        try {
          const job = await request(options, `/v1/videos/${encodeURIComponent(id)}`)
          if (!["completed", "failed"].includes(job.status)) continue
          let result = job
          if (job.status === "completed") result = await downloadResult(options, id)
          const actual = result.actual
          const message = job.status === "completed"
            ? `${result.project || "WAN"} / ${result.shot || id}\n${actual?.duration ?? "?"} 秒 · ${actual?.width ?? "?"}×${actual?.height ?? "?"} · ${actual?.fps ?? "?"} fps\n${result.prompt || ""}\n输入：${(result.input_frames || []).map(frame => frame.name || frame.role).join(", ")}\n预览：${result.preview_url}`
            : `${id}: ${job.error || "生成失败"}`
          await input.client.tui.showToast({ body: { title: job.status === "completed" ? "WAN 视频已完成" : "WAN 视频失败", message, variant: job.status === "completed" ? "success" : "error" } })
          pending.delete(id)
          await persist()
        } catch (error) {
          if (error.status === 404) { pending.delete(id); await persist() }
          // A transient service/download error leaves the ID queued for the next pass.
        }
      }
    } finally { busy = false; schedule() }
  }
  return async id => { await ready; if (typeof id !== "string" || !id) return; pending.add(id); await persist(); schedule() }
}

const PROVIDER = "modelscope-wan"
const exec = promisify(execFile)
const CONNECT_ACCOUNT_ENV = "WAN_MAGPIE_CONNECT_ACCOUNT"

async function saveMagpieAccount(account, options, signal) {
  // Use Magpie's public login path so it deduplicates accounts and refreshes
  // Providers/Usage. Never edit its auth files or call auth.set across scopes.
  const candidates = options.magpieExecutable ? [options.magpieExecutable] : [
    "magpie", ...(process.platform === "darwin" ? [
      "/Applications/magpie.app/Contents/MacOS/magpie",
      join(homedir(), "Applications/magpie.app/Contents/MacOS/magpie"),
    ] : []),
  ]
  for (const executable of candidates) {
    try {
      await exec(executable, ["plugin", "login", PROVIDER, "1"], {
        env: { ...process.env, [CONNECT_ACCOUNT_ENV]: account },
        signal, timeout: 30_000, maxBuffer: 1024 * 1024, windowsHide: true,
      })
      return
    } catch (error) {
      if (error.code === "ENOENT") continue
      // Child output may contain other providers' configuration. Do not expose it.
      throw new Error(`账号 ${account} 同步到 Magpie 失败；已同步的账号会保留，请重试。`)
    }
  }
  throw new Error("找不到 Magpie 命令；请将 magpie 加入 PATH，或配置插件的 magpieExecutable 路径。")
}

export const MagpieWanTools = async (input = {}, options = {}) => {
  const watch = completionWatcher(input, options)
  const icon = `data:image/png;base64,${await readFile(fileURLToPath(new URL("./assets/modelscope-icon.png", import.meta.url)), "base64")}`
  const connectAccount = async (account, signal) => {
    if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(account)) return { type: "failed", error: "请填写 WAN_MODELSCOPE_ACCOUNTS 中的账号 ID" }
    try {
      const profile = await request(options, `/v1/accounts/${encodeURIComponent(account)}`, { signal })
      if (profile.status !== "connected" || profile.stale) return { type: "failed",
        error: profile.session_error || (profile.stale ? "账号状态尚未刷新，请检查 WAN 服务和浏览器登录" : "ModelScope 账号暂不可用") }
      return { type: "success", key: account, metadata: { email: `${profile.label} · ${account}`, wan_account: account } }
    } catch (error) { return { type: "failed", error: error.message } }
  }
  const syncAccounts = async (_inputs, { signal } = {}) => {
    // A child CLI invocation saves one account via the normal host callback.
    // It must never launch another sync or inherit a stale list of accounts.
    const childAccount = process.env[CONNECT_ACCOUNT_ENV]
    if (childAccount) return connectAccount(childAccount, signal)
    const deadline = AbortSignal.timeout(45_000)
    const abort = AbortSignal.any([signal, deadline].filter(Boolean))
    try {
      let profiles
      const until = Date.now() + (options.loginWaitMs ?? 20_000)
      do {
        profiles = (await request(options, "/v1/accounts", { signal: abort })).accounts || []
        if (profiles.some(p => p.status === "connected" && !p.stale) || Date.now() >= until) break
        await delay(1000, undefined, { signal: abort })
      } while (true)
      const ready = profiles.filter(p => p.status === "connected" && !p.stale)
      if (!ready.length) return { type: "failed", error: profiles.length
        ? "尚未检测到有效登录。请在 WAN 服务配置的浏览器窗口登录 ModelScope 后，再点登录同步。" + (profiles[0].session_error || "")
        : "WAN 服务没有配置账号，请先配置 WAN_MODELSCOPE_ACCOUNTS。" }
      for (const profile of ready) {
        if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(profile.id)) throw new Error("WAN 服务返回了无效的账号 ID")
      }
      // Save every account through the CLI, including the first. The outer
      // success is then deduplicated by Magpie and refreshes the open UI.
      for (const profile of ready) await saveMagpieAccount(profile.id, options, abort)
      abort.throwIfAborted()
      const result = await connectAccount(ready[0].id, abort)
      if (result.type === "success") {
        const skipped = profiles.filter(p => !ready.includes(p)).map(p => p.id)
        await input.client?.tui?.showToast?.({ body: {
          title: "ModelScope 账号已同步", variant: "success",
          message: `已同步 ${ready.length} 个账号，可在供应商和用量中查看魔粒余额。` +
            (skipped.length ? `未同步：${skipped.join("、")}，请登录后重试。` : ""),
        } })?.catch(() => {})
      }
      return result
    } catch (error) {
      return { type: "failed", error: abort.aborted ? "登录同步已取消或超时；已同步的账号会保留，可重试。" : error.message }
    }
  }
  return {
    config: async cfg => {
      cfg.provider ??= {}
      cfg.provider[PROVIDER] ??= { name: "ModelScope WAN", models: {} }
    },
    auth: {
      provider: PROVIDER,
      icon,
      methods: [{ type: "api", keyless: true, label: "登录并同步浏览器账号", authorize: syncAccounts }],
      async usage(getAuth, _provider, { signal } = {}) {
        const auth = await getAuth()
        const account = String(auth?.key || "").trim()
        if (!account) return { error: "请先连接本机 ModelScope 账号", windows: [] }
        const profile = await request(options, `/v1/accounts/${encodeURIComponent(account)}`, { signal })
        const user = `${profile.label || account} · ${account}`
        if (profile.status !== "connected" || profile.stale) return { user,
          signIn: profile.session_error ? "expired" : undefined,
          error: profile.session_error || (profile.stale ? "账号状态尚未刷新，请检查 WAN 服务和浏览器登录" : "ModelScope 账号暂不可用"), windows: [] }
        if (typeof profile.balance !== "number" || !Number.isFinite(profile.balance)) return { user, error: profile.balance_error || "魔粒余额暂不可用", windows: [] }
        return { user, plan: "ModelScope WAN", balance: `${profile.balance} 魔粒`,
          windows: [{ name: "可用魔粒", balance: profile.balance, unit: "魔粒", display: `${profile.balance} 魔粒` }] }
      },
    },
    provider: { id: PROVIDER, models: async () => ({}) },
    tool: {
      wan_generate_flf2v: submitTool(options, true, "wan2.1-flf2v-14b", watch),
      wan_generate_i2v: submitTool(options, false, "wan2.2-i2v-a14b", watch),
      wan_get_video_status: statusTool(options),
      wan_get_video_result: resultTool(options),
      wan_list_video_jobs: listTool(options),
      wan_get_config: tool({ description: "Show allowed input directories and each local ModelScope account's connection and magicube balance.", args: {},
        async execute(_args, ctx = {}) { const [media, accounts] = await Promise.all([
          request(options, "/v1/media/roots", { signal: ctx.abort }), request(options, "/v1/accounts", { signal: ctx.abort }),
        ]); return { ...media, ...accounts, usage_unit: "魔粒" } } }),
      wan_prepare_frames: tool({ description: "Validate frame paths before submitting; explicitly enable copying images outside allowed directories.",
        args: { first_frame: tool.schema.string().min(1), last_frame: tool.schema.string().optional(), copy_to_media_root: tool.schema.boolean().optional() },
        execute: (args, ctx = {}) => prepareFrames(args, options, ctx.abort) }),
    },
  }
}

// Export the same function so Magpie's loader runs it only once.
export default MagpieWanTools
