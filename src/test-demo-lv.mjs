// node src/test-demo-lv.mjs [shots dir] — panel 15 (LOCAL VOL) of the demo in headless
// Chromium on its software GPU, driven like a user: the expiry slider over 3, 7, 28 and
// 98 days, and both model-risk chips. At each expiry the GPU reprices up to 16
// liquid SPY quotes under local vol and turns them back into implied vols: the worst
// round-trip error must stay under 0.1 vol points and within 0.01 vol points of
// Python's on the same grid. Needs npm run build and playwright, as src/test-gpu.mjs
// does. With a directory, saves screenshots there.
import { preview } from "vite";
import { chromium } from "playwright";
const SHOTS = process.argv[2];
const server = await preview({ logLevel: "warn", preview: { port: 4177, host: "127.0.0.1" } });
const url = `${server.resolvedUrls.local[0]}`;
const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1180, height: 900 } });
page.on("pageerror", (e) => console.error(`page error: ${e.message}`));
page.on("console", (m) => m.type() === "error" && console.error(`console: ${m.text()}`));
await page.goto(url);
await page.waitForFunction(() => window.localVol && window.localVol.res, null, { timeout: 600e3, polling: 500 });
const panel = page.locator("#lvPanel");
await panel.scrollIntoViewIfNeeded();
const settle = () =>
  page.waitForFunction(
    () => {
      const t = window.localVol;
      return !t.busy && (t.failure || (t.res && t.res.seq === t.asked));
    },
    null,
    { timeout: 600e3, polling: 200 },
  );
let fails = 0;
const row = async (what) => {
  await settle();
  const failure = await page.evaluate(() => window.localVol.failure);
  if (failure) {
    console.log(`FAIL ${what}: the reprice threw: ${failure.message}`);
    fails++;
    return;
  }
  const r = await page.evaluate(() => {
    const t = window.localVol, r = t.res, ex = t.data.expiries[r.e];
    return { name: ex.name, days: Math.round(ex.T * 365), n: r.pts.length, worst: r.worst, vsPy: r.vsPy, ms: r.ms, steps: r.steps, spots: r.n };
  });
  const ok = r.n > 0 && r.worst < 0.1 && r.vsPy < 0.01;
  if (!ok) fails++;
  console.log(
    `${ok ? "PASS" : "FAIL"} ${what.padEnd(14)} ${r.name} (${String(r.days).padStart(3)}d)  ${String(r.n).padStart(2)} quotes  ` +
      `worst round trip ${r.worst.toFixed(3)} vol pts  gpu vs python ${r.vsPy.toFixed(4)}  ${r.steps} steps x ${r.spots} spots ${r.ms.toFixed(0)} ms`,
  );
};
const shot = async (name) => SHOTS && panel.screenshot({ path: `${SHOTS}/${name}.png` });
await row("start");
await shot("lv-start");
for (const e of [0, 4, 18]) {
  await page.evaluate((e) => window.localVol.setExpiry(e), e);
  await row(`expiry ${e}`);
}
await shot("lv-98d");
for (const b of [1, 0]) {
  await page.click(`[data-lvr="${b}"]`);
  const foot = await page.$eval("#lvRiskFoot", (e) => e.textContent);
  const ok = /\$\d+\.\d\d APART/.test(foot);
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} model-risk chip ${b}: ${foot}`);
  if (b === 1) await shot("lv-risk-342d");
}
console.log(fails ? `${fails} FAILED` : "all demo checks PASS");
await browser.close();
await server.close();
process.exit(fails ? 1 : 0);
