import { tool } from "@opencode-ai/plugin"
import { createWriteStream } from "node:fs"
import { mkdir, mkdtemp, rename, rm } from "node:fs/promises"
import { homedir } from "node:os"
import { isAbsolute, join } from "node:path"
import { Readable } from "node:stream"
import { pipeline } from "node:stream/promises"
import { pathToFileURL } from "node:url"

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

async function request(options, path, init = {}) {
  const response = await fetch(`${serviceUrl(options)}${path}`, {
    ...init,
    headers: { ...headers(options), ...(init.headers || {}) },
    redirect: "error",
    signal: init.signal,
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = payload?.error?.message || `WAN service returned HTTP ${response.status}`
    throw new Error(message)
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
    idempotency_key: tool.schema.string().optional().describe("Optional key to return the same job for retries of this exact request."),
  }
  if (includeLastFrame) {
    fields.last_frame = tool.schema.string().min(1).describe("Absolute local path to the last frame image.")
  }
  return fields
}

function submitTool(options, includeLastFrame, defaultModel) {
  return tool({
    description: includeLastFrame
      ? "Submit a WAN first-frame-to-last-frame video job. Returns a durable job id immediately; use wan_get_video_status to check progress."
      : "Submit a WAN first-frame-to-video job. Returns a durable job id immediately; use wan_get_video_status to check progress.",
    args: generateArgs(includeLastFrame, defaultModel),
    async execute(args, ctx) {
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
        idempotency_key: args.idempotency_key,
      }
      return await request(options, "/v1/videos", {
        method: "POST",
        headers: args.idempotency_key ? { "Idempotency-Key": args.idempotency_key } : {},
        body: JSON.stringify(body),
        signal: ctx.abort,
      })
    },
  })
}

function statusTool(options) {
  return tool({
    description: "Check a WAN job without waiting for generation. Completed jobs include a local content_url.",
    args: { job_id: tool.schema.string().min(1).describe("Job id returned by a WAN submit tool.") },
    async execute({ job_id }, ctx) {
      return await request(options, `/v1/videos/${encodeURIComponent(job_id)}`, { signal: ctx.abort })
    },
  })
}

function resultTool(options) {
  return tool({
    description: "Download a completed WAN video with service authentication and return an absolute local output_path and file_url. If still running, returns status.",
    args: { job_id: tool.schema.string().min(1).describe("Job id returned by a WAN submit tool.") },
    async execute({ job_id }, ctx) {
      const signal = AbortSignal.any([ctx.abort, AbortSignal.timeout(50_000)].filter(Boolean))
      const path = `/v1/videos/${encodeURIComponent(job_id)}`
      const result = await request(options, path, { signal })
      if (result.status !== "completed") return result
      const root = options?.outputDir || join(homedir(), ".magpie-wan", "results")
      if (!isAbsolute(root)) throw new Error("WAN outputDir must be an absolute local directory")
      await mkdir(root, { recursive: true, mode: 0o700 })
      const directory = await mkdtemp(join(root, "wan-"))
      const output = join(directory, "video.mp4")
      try {
        // Construct the content route ourselves; never forward a token to a URL from a response.
        const response = await fetch(`${serviceUrl(options)}${path}/content`, {
          headers: headers(options), signal, redirect: "error",
        })
        if (!response.ok || !response.body) {
          await response.body?.cancel()
          throw new Error(`WAN content download returned HTTP ${response.status}`)
        }
        await pipeline(Readable.fromWeb(response.body),
          createWriteStream(`${output}.partial`, { flags: "wx", mode: 0o600 }), { signal })
        signal.throwIfAborted()
        await rename(`${output}.partial`, output)
        return {
          id: result.id,
          status: result.status,
          output_path: output,
          file_url: pathToFileURL(output).href,
        }
      } catch (error) {
        await rm(directory, { recursive: true, force: true })
        throw error
      }
    },
  })
}

export const MagpieWanTools = async (_input, options) => ({
  tool: {
    wan_generate_flf2v: submitTool(options, true, "wan2.1-flf2v-14b"),
    wan_generate_i2v: submitTool(options, false, "wan2.2-i2v-a14b"),
    wan_get_video_status: statusTool(options),
    wan_get_video_result: resultTool(options),
  },
})
