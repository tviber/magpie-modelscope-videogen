import assert from "node:assert/strict"
import { mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises"
import { createServer } from "node:http"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import test from "node:test"

const { MagpieWanTools, default: defaultPlugin } = await import(process.env.WAN_TEST_PLUGIN || "../index.mjs")
// Capture before MockTimers is enabled: filesystem polling needs real time.
const realTimeout = globalThis.setTimeout

async function fixture(t, handler, options = {}) {
  const outputDir = await mkdtemp(join(tmpdir(), "wan-test-"))
  const server = createServer((req, res) => {
    if (req.url === "/v1/accounts" && !options.testAccounts) return res.end('{"accounts":[]}')
    return handler(req, res)
  })
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve))
  t.after(async () => {
    server.closeAllConnections()
    await new Promise(resolve => server.close(resolve))
    await rm(outputDir, { recursive: true, force: true })
  })
  const opts = { serverUrl: `http://127.0.0.1:${server.address().port}`, outputDir, ...options }
  const plugin = await MagpieWanTools({}, opts)
  assert.equal(Object.keys(plugin.tool).length, 7)
  return { tool: plugin.tool.wan_get_video_result, outputDir, tools: plugin.tool, plugin, options: opts }
}

test("result download authenticates and returns a readable local artifact without a token URL", async t => {
  const seen = []
  const { tool } = await fixture(t, (req, res) => {
    seen.push({ url: req.url, auth: req.headers.authorization })
    if (req.headers.authorization !== "Bearer private-test-token") {
      res.writeHead(401).end()
    } else if (req.url.endsWith("/content")) {
      res.writeHead(200, { "content-type": "video/mp4" }).end("video fixture")
    } else {
      res.setHeader("content-type", "application/json")
      res.end(JSON.stringify({ id: "job", status: "completed", content_url: "https://evil.invalid/steal" }))
    }
  }, { token: "private-test-token" })
  const result = await tool.execute({ job_id: "job" }, { abort: new AbortController().signal })
  assert.equal(await readFile(result.output_path, "utf8"), "video fixture")
  assert.equal(fileURLToPath(result.file_url), result.output_path)
  assert.equal(seen.length, 2)
  assert.equal(seen[1].url, "/v1/videos/job/content")
  assert.ok(seen.every(item => item.auth === "Bearer private-test-token"))
  assert.ok(!JSON.stringify(result).includes("private-test-token"))
  assert.ok(!JSON.stringify(result).includes("evil.invalid"))
})

test("unfinished result returns status without creating an artifact", async t => {
  const { tool, outputDir } = await fixture(t, (_req, res) => res.end('{"status":"in_progress"}'))
  assert.equal((await tool.execute({ job_id: "job" }, {})).status, "in_progress")
  assert.deepEqual(await readdir(outputDir), [])
})

test("submit checks the allowed roots before posting and can copy an outside frame", async t => {
  const sourceDir = await mkdtemp(join(tmpdir(), "wan-outside-"))
  const source = join(sourceDir, "frame.png")
  await writeFile(source, Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aC1sAAAAASUVORK5CYII=", "base64"))
  t.after(() => rm(sourceDir, { recursive: true, force: true }))
  const allowed = await mkdtemp(join(tmpdir(), "wan-media-"))
  t.after(() => rm(allowed, { recursive: true, force: true }))
  let submitted
  const { tools } = await fixture(t, async (req, res) => {
    if (req.url === "/v1/media/roots") return res.end(JSON.stringify({ roots: [allowed] }))
    if (req.url === "/v1/frames/validate") return res.end('{"valid":true,"frames":{"first_frame":{"width":320,"height":180}}}')
    if (req.method === "POST") {
      submitted = JSON.parse(await new Promise(resolve => { let body = ""; req.on("data", part => body += part); req.on("end", () => resolve(body)) }))
      return res.end(JSON.stringify({ id: "job", status: "queued" }))
    }
    res.statusCode = 404
    res.end()
  })
  const args = { prompt: "A camera move", first_frame: source }
  await assert.rejects(tools.wan_generate_i2v.execute(args, { abort: new AbortController().signal }), /Allowed directories/)
  assert.equal(submitted, undefined)
  await tools.wan_generate_i2v.execute({ ...args, copy_to_media_root: true }, { abort: new AbortController().signal })
  assert.ok(submitted.first_frame.includes("wan-media-"))
  assert.ok(submitted.first_frame !== source)
})

test("one plugin export and independent account labels and magicube wallets", async t => {
  assert.equal(defaultPlugin, MagpieWanTools)
  const profiles = [
    { id: "a", label: "Alice", status: "connected", balance: 45, stale: false },
    { id: "b", label: "Bob", status: "connected", balance: 0, stale: false },
  ]
  const { plugin, options } = await fixture(t, (req, res) => {
    if (req.url === "/v1/accounts") return res.end(JSON.stringify({ accounts: profiles }))
    res.end(JSON.stringify(profiles.find(p => req.url.endsWith(`/${p.id}`))))
  }, { testAccounts: true })
  const config = {}
  await plugin.config(config)
  assert.deepEqual(config.provider["modelscope-wan"].models, {})
  assert.equal(plugin.auth.methods.length, 1)
  assert.match(plugin.auth.icon, /^data:image\/png;base64,/)
  for (let i = 0; i < profiles.length; i++) {
    process.env.WAN_MAGPIE_CONNECT_ACCOUNT = profiles[i].id
    const login = await (await plugin.auth.methods[0].authorize()).callback()
    delete process.env.WAN_MAGPIE_CONNECT_ACCOUNT
    assert.equal(login.key, profiles[i].id)
    assert.equal(login.metadata.email, `${profiles[i].label} · ${profiles[i].id}`)
    const usage = await plugin.auth.usage(async () => ({ type: "api", key: login.key }))
    assert.equal(usage.windows[0].balance, profiles[i].balance)
    assert.equal(usage.windows[0].unit, "魔粒")
    assert.equal(usage.windows[0].used, undefined)
  }
  process.env.WAN_MAGPIE_CONNECT_ACCOUNT = "a"
  t.after(() => { delete process.env.WAN_MAGPIE_CONNECT_ACCOUNT })
  profiles[0].stale = true
  assert.equal((await (await plugin.auth.methods[0].authorize()).callback()).type, "failed")
  assert.deepEqual((await plugin.auth.usage(async () => ({ key: "a" }))).windows, [])
  profiles[1].balance = null
  const unknown = await plugin.auth.usage(async () => ({ key: "b" }))
  assert.deepEqual(unknown.windows, [])
  assert.ok(unknown.error)
})

test("account connection and usage explain rejected browser sessions", async t => {
  const profile = { id: "a", label: "Alice", status: "unavailable", balance: null, stale: false,
    session_error: "ModelScope rejected this browser session (HTTP 401). Sign in again." }
  const { plugin, options } = await fixture(t, (req, res) => {
    if (req.url === "/v1/accounts") return res.end(JSON.stringify({ accounts: [profile] }))
    res.end(JSON.stringify(profile))
  }, { testAccounts: true })
  process.env.WAN_MAGPIE_CONNECT_ACCOUNT = "a"
  t.after(() => { delete process.env.WAN_MAGPIE_CONNECT_ACCOUNT })
  const connect = await (await plugin.auth.methods[0].authorize()).callback()
  assert.equal(connect.type, "failed")
  assert.match(connect.error, /HTTP 401/)
  const usage = await plugin.auth.usage(async () => ({ key: "a" }))
  assert.equal(usage.user, "Alice · a")
  assert.equal(usage.signIn, "expired")
  assert.match(usage.error, /HTTP 401/)
  assert.deepEqual(usage.windows, [])
})

test("native Magpie login only saves the matching verified account after callback", async t => {
  const id = "a".repeat(48)
  let state = "waiting", polls = 0
  const { plugin } = await fixture(t, async (req, res) => {
    if (req.method === "POST" && req.url === "/v1/logins") {
      let body = ""
      for await (const chunk of req) body += chunk
      assert.deepEqual(JSON.parse(body), {})
      return res.end(JSON.stringify({ id, account: "account-3" }))
    }
    if (req.url === `/v1/logins/${id}`) { polls++; return res.end(JSON.stringify({ id, account: "account-3", state })) }
    if (req.url === "/v1/accounts/account-3") return res.end(JSON.stringify({
      id: "account-3", label: "Alice", status: "connected", stale: false,
    }))
    res.end("{}")
  }, { testAccounts: true })
  const method = plugin.auth.methods[0]
  assert.equal(plugin.auth.methods.length, 1)
  assert.equal(method.label, "登录新账号")
  assert.equal(method.prompts, undefined)
  assert.equal(method.ask, undefined)
  assert.equal(method.type, "oauth")
  const flow = await method.authorize({ account: "ignored-old-input" })
  assert.equal(flow.method, "auto")
  assert.equal(flow.url, "")
  assert.equal(flow.key, undefined)
  assert.equal(polls, 0)
  state = "completed"
  const result = await flow.callback()
  assert.equal(result.type, "success")
  assert.equal(result.key, "account-3")
  assert.equal(result.metadata.wan_account, "account-3")
  assert.deepEqual(await flow.callback(), result)
  assert.equal(polls, 1)
})

for (const scenario of ["cancelled", "mismatch", "unverified"]) {
  test(`native Magpie login rejects ${scenario} and cancels its own task`, async t => {
    const id = "c".repeat(48)
    let cancelled = 0
    const { plugin } = await fixture(t, (req, res) => {
      if (req.url.endsWith("/cancel")) { cancelled++; return res.end("{}") }
      if (req.url === "/v1/logins") return res.end(JSON.stringify({ id, account: "a" }))
      if (req.url === `/v1/logins/${id}`) return res.end(JSON.stringify({ id,
        account: scenario === "mismatch" ? "b" : "a", state: scenario === "cancelled" ? "cancelled" : "completed" }))
      res.end(JSON.stringify({ id: "a", status: "unavailable", stale: true }))
    }, { testAccounts: true })
    assert.equal((await (await plugin.auth.methods[0].authorize()).callback()).type, "failed")
    assert.equal(cancelled, 1)
  })
}

test("result packages escaped preview, frame provenance, specifications and manifest", async t => {
  const seen = []
  const { tool } = await fixture(t, (req, res) => {
    seen.push(req.url)
    if (req.url.endsWith("/content") || req.url.includes("/frames/")) return res.end("media fixture")
    res.end(JSON.stringify({ id: "shot-1", status: "completed", project: "MV", shot: "Opening",
      prompt: '<script>alert("unsafe")</script>', requested: { seconds: 8, size: "1280x720" },
      actual: { duration: 5.06, width: 1280, height: 736, fps: 16 }, spec_warnings: ["Duration mismatch"],
      estimate: 45, estimate_unit: "魔粒",
      input_frames: [{ role: "first_frame", name: "opening.png", width: 1280, height: 720, preview_url: "https://evil.invalid/frame" }],
    }))
  })
  const result = await tool.execute({ job_id: "shot-1" })
  assert.equal(result.actual.duration, 5.06)
  assert.equal(result.estimate_unit, "魔粒")
  assert.equal(result.input_frames[0].name, "opening.png")
  assert.ok(seen.includes("/v1/videos/shot-1/frames/first_frame"))
  assert.deepEqual(JSON.parse(await readFile(result.manifest_path, "utf8")), JSON.parse(JSON.stringify(result)))
  const html = await readFile(result.preview_path, "utf8")
  assert.ok(html.includes("&lt;script&gt;"))
  assert.ok(!html.includes("<script>"))
  assert.ok(html.includes('src="video.mp4"'))
  assert.ok(!JSON.stringify(result).includes("evil.invalid"))
})

test("symlinks do not bypass input roots and validation failures clean staged copies", async t => {
  if (process.platform === "win32") return t.skip("Creating symlinks requires Windows developer mode")
  const allowed = await mkdtemp(join(tmpdir(), "wan-roots-"))
  const outside = await mkdtemp(join(tmpdir(), "wan-outside-"))
  t.after(async () => { await rm(allowed, { recursive: true, force: true }); await rm(outside, { recursive: true, force: true }) })
  const source = join(outside, "frame.png")
  await writeFile(source, "invalid image")
  const linked = join(allowed, "linked.png")
  await symlink(source, linked)
  const { tools } = await fixture(t, (req, res) => {
    if (req.url === "/v1/media/roots") return res.end(JSON.stringify({ roots: [allowed] }))
    res.writeHead(400).end('{"error":{"message":"frame file is not a readable image"}}')
  })
  await assert.rejects(tools.wan_prepare_frames.execute({ first_frame: linked }), /outside WAN_MEDIA_ROOTS/)
  await assert.rejects(tools.wan_prepare_frames.execute({ first_frame: linked, copy_to_media_root: true }), /readable image/)
  assert.deepEqual(await readdir(allowed), ["linked.png"])
})

test("authentication failure leaves no artifact", async t => {
  const { tool, outputDir } = await fixture(t, (req, res) => {
    if (req.url.endsWith("/content")) res.writeHead(401).end()
    else res.end('{"status":"completed"}')
  })
  await assert.rejects(tool.execute({ job_id: "job" }, {}), /HTTP 401/)
  assert.deepEqual(await readdir(outputDir), [])
})

test("redirects cannot receive the service token", async t => {
  let redirected = false
  const { tool, outputDir } = await fixture(t, (req, res) => {
    if (req.url === "/steal") {
      redirected = true
      res.end("stolen")
    } else if (req.url.endsWith("/content")) {
      res.writeHead(307, { location: "/steal" }).end()
    } else res.end('{"status":"completed"}')
  }, { token: "private-test-token" })
  await assert.rejects(tool.execute({ job_id: "job" }, {}))
  assert.equal(redirected, false)
  assert.deepEqual(await readdir(outputDir), [])
})

test("abort removes a partially downloaded video", async t => {
  const controller = new AbortController()
  const { tool, outputDir } = await fixture(t, (req, res) => {
    if (req.url.endsWith("/content")) {
      res.writeHead(200, { "content-type": "video/mp4" })
      res.write("partial")
      setTimeout(() => controller.abort(), 30)
    } else res.end('{"status":"completed"}')
  })
  await assert.rejects(tool.execute({ job_id: "job" }, { abort: controller.signal }))
  assert.deepEqual(await readdir(outputDir), [])
})

test("completion watcher resumes persisted tasks, downloads once and emits useful notification", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] })
  const notifications = []
  const { createHash } = await import("node:crypto")
  const { outputDir, options: opts } = await fixture(t, (req, res) => {
    if (req.url.endsWith("/content")) return res.end("video fixture")
    res.end(JSON.stringify({ id: "watched", status: "completed", project: "MV", shot: "Opening", prompt: "A steady camera",
      actual: { duration: 5.06, width: 1280, height: 736, fps: 16 }, input_frames: [] }))
  })
  const state = join(outputDir, `watch-${createHash("sha256").update(opts.serverUrl).digest("hex").slice(0, 24)}.json`)
  await writeFile(state, JSON.stringify(["watched"]))
  await MagpieWanTools({ client: { tui: { showToast: async payload => { notifications.push(payload.body) } } } }, opts)
  // Filesystem completion is not tied to an event-loop iteration count, especially
  // on Windows. Advance the watch timer until notified, then wait for the atomic
  // state write, with a real deadline (performance is not mocked).
  const deadline = performance.now() + 5_000
  let saved
  while (performance.now() < deadline) {
    // Do not continuously reopen the target while Windows is replacing it;
    // an open reader can prevent the atomic rename there.
    await new Promise(resolve => realTimeout(resolve, 25))
    if (notifications.length === 0) t.mock.timers.tick(15_000)
    saved = JSON.parse(await readFile(state, "utf8"))
    if (notifications.length && saved.length === 0) break
  }
  assert.equal(notifications.length, 1)
  assert.match(notifications[0].message, /5.06 秒 · 1280×736 · 16 fps/)
  assert.match(notifications[0].message, /A steady camera/)
  assert.match(notifications[0].message, /preview.html/)
  assert.deepEqual(saved, [], "completed task must be removed from persisted watch state within 5 seconds")
})

test("native login uses callback cancellation, not the short-lived authorize signal", async t => {
  const id = "e".repeat(48)
  let cancelled = 0
  const { plugin } = await fixture(t, (req, res) => {
    if (req.url.endsWith('/cancel')) { cancelled++; return res.end('{}') }
    if (req.url === '/v1/logins') return res.end(JSON.stringify({ id, account: 'a' }))
    if (req.url === `/v1/logins/${id}`) return res.end(JSON.stringify({ id, account: 'a', state: 'completed' }))
    res.end(JSON.stringify({ id: 'a', label: 'Alice', status: 'connected', stale: false }))
  }, { testAccounts: true })
  const authorizeController = new AbortController()
  const flow = await plugin.auth.methods[0].authorize({}, { signal: authorizeController.signal })
  assert.equal(flow.opened, true)
  assert.equal(flow.url, '')
  authorizeController.abort()
  assert.equal((await flow.callback()).type, 'success')
  assert.equal(cancelled, 0)
  const next = await plugin.auth.methods[0].authorize()
  const callbackController = new AbortController()
  callbackController.abort()
  assert.equal((await next.callback(undefined, { signal: callbackController.signal })).type, 'failed')
  assert.equal(cancelled, 1)
})

test("CLI account synchronization cannot launch a new browser through native login", async t => {
  const previous = process.env.WAN_MAGPIE_CONNECT_ACCOUNT
  process.env.WAN_MAGPIE_CONNECT_ACCOUNT = 'existing'
  t.after(() => {
    if (previous === undefined) delete process.env.WAN_MAGPIE_CONNECT_ACCOUNT
    else process.env.WAN_MAGPIE_CONNECT_ACCOUNT = previous
  })
  let launches = 0
  const { plugin } = await fixture(t, (req, res) => {
    if (req.url === '/v1/logins') launches++
    res.end(JSON.stringify({ id: 'existing', status: 'connected', stale: false, label: 'Existing' }))
  }, { testAccounts: true })
  const flow = await plugin.auth.methods[0].authorize()
  assert.equal((await flow.callback()).key, 'existing')
  assert.equal(launches, 0)
})

test("repeat login saves verified original account while retaining task correlation", async t => {
  const id = 'f'.repeat(48)
  const { plugin } = await fixture(t, (req, res) => {
    if (req.url === '/v1/logins') return res.end(JSON.stringify({ id, account: 'new' }))
    if (req.url === `/v1/logins/${id}`) return res.end(JSON.stringify({ id, account: 'new', result_account: 'old', state: 'completed' }))
    assert.equal(req.url, '/v1/accounts/old')
    res.end(JSON.stringify({ id: 'old', label: 'Existing', status: 'connected', stale: false }))
  }, { testAccounts: true })
  const result = await (await plugin.auth.methods[0].authorize()).callback()
  assert.equal(result.type, 'success')
  assert.equal(result.key, 'old')
})
