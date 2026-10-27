// node src/test-gpu.mjs [section] — /check.html in headless Chromium on its software
// WebGPU adapter (SwiftShader), for machines without a GPU such as CI runners. It runs
// the same WGSL shaders in 32-bit floats against the same Python fixtures as on a real
// GPU, only slower. section: fixtures, fft, kernels, chain, odds, 2d, lv or diffusion (default: all eight).
// Needs the built pages (npm run build) and playwright with its Chromium
// (npm i --no-save playwright && npx playwright install chromium).
import { preview } from "vite";
import { chromium } from "playwright";

const section = process.argv[2];
const server = await preview({ logLevel: "warn", preview: { port: 4173, host: "127.0.0.1" } });
const url = `${server.resolvedUrls.local[0]}check.html${section ? `?only=${section}` : ""}`;
const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error(`page error: ${e.message}`));
console.log(`${url}\n`);
await page.goto(url);
// check.js sets the title to PASS, FAIL or ERROR when it is done
let title = "";
try {
  await page.waitForFunction(() => ["PASS", "FAIL", "ERROR"].includes(document.title), null, { timeout: 20 * 60e3, polling: 1000 });
  title = await page.title();
} catch {
  title = "TIMEOUT";
}
console.log(await page.$eval("#out", (e) => e.textContent));
console.log(title);
await browser.close();
await server.close();
process.exit(title === "PASS" ? 0 : 1);
