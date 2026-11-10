import assert from "node:assert/strict"
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises"
import { createServer } from "node:http"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import test from "node:test"

const { MagpieWanTools } = await import(process.env.WAN_TEST_PLUGIN || "../index.mjs")

async function fixture(t, handler, options = {}) {
  const outputDir = await mkdtemp(join(tmpdir(), "wan-test-"))
  const server = createServer(handler)
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve))
  t.after(async () => {
    server.closeAllConnections()
    await new Promise(resolve => server.close(resolve))
    await rm(outputDir, { recursive: true, force: true })
  })
  const plugin = await MagpieWanTools({}, {
    serverUrl: `http://127.0.0.1:${server.address().port}`, outputDir, ...options,
  })
  assert.equal(Object.keys(plugin.tool).length, 4)
  return { tool: plugin.tool.wan_get_video_result, outputDir }
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
