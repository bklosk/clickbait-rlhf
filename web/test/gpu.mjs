// Node harness: expose Dawn's WebGPU as the browser globals the library expects.
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

// Dawn tears the device down if its GPU instance is garbage collected, so keep it alive.
const gpu = create([]);

export async function getDevice() {
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  device.addEventListener?.("uncapturederror", (e) => { console.error("WebGPU error:", e.error.message); process.exitCode = 1; });
  return device;
}
