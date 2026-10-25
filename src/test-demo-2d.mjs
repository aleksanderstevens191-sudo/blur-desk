// node src/test-demo-2d.mjs [shots dir] — panel 14 (TWO STOCKS) of the demo in headless
// Chromium on its software GPU, driven like a user: presets, the correlation sweep,
// wide vols, both GPU methods, and a painted payoff. Each state must price the spot
// within 1e-4 of its closed form (Margrabe, Stulz, the exact spread integral) and the
// GPU within 1e-4 of the 64-bit blur at 9 points of the patch. Needs npm run build
// and playwright, as src/test-gpu.mjs does. With a directory, saves screenshots there.
import { preview } from "vite";
import { chromium } from "playwright";
const SHOTS = process.argv[2];
const server = await preview({ logLevel: "warn", preview: { port: 4176, host: "127.0.0.1" } });
const url = `${server.resolvedUrls.local[0]}`;
const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1180, height: 900 } });
page.on("pageerror", (e) => console.error(`page error: ${e.message}`));
page.on("console", (m) => m.type() === "error" && console.error(`console: ${m.text()}`));
await page.goto(url);
await page.waitForFunction(() => window.twoStock && window.twoStock.res, null, { timeout: 600e3, polling: 500 });
const panel = page.locator("#tsPanel");
await panel.scrollIntoViewIfNeeded();
const set = async (id, v) =>
  page.$eval(
    `#${id}`,
    (e, v) => {
      e.value = v;
      e.dispatchEvent(new Event("input"));
    },
    v,
  );
// wait until the panel has answered the latest request, or failed on it
const settle = () =>
  page.waitForFunction(
    () => {
      const t = window.twoStock;
      return !t.busy && (t.failure || (t.res && t.res.seq === t.asked));
    },
    null,
    { timeout: 600e3, polling: 200 },
  );
const row = async (what) => {
  await settle();
  const failure = await page.evaluate(() => window.twoStock.failure);
  if (failure) {
    console.log(`FAIL ${what}: the reprice threw: ${failure.message}`);
    return false;
  }
  const r = await page.evaluate(() => {
    const t = window.twoStock,
      r = t.res;
    return { st: r.state, spot: r.spot, cf: r.cf, worst: r.worst, ms: r.ms, N: r.N, parts: r.parts, taps: r.taps, method: r.method };
  });
  const err = r.cf ? Math.abs(r.spot - r.cf.v) : null;
  const k = r.st.preset === "best" ? ` K=${r.st.K.best}` : r.st.preset === "spread" ? ` K=${r.st.K.spread}` : "";
  const ok = (err === null || err < 1e-4) && r.worst < 1e-4;
  console.log(
    `${ok ? "PASS" : "FAIL"} ${what.padEnd(26)} ${r.st.preset.padEnd(8)}${k.padEnd(7)} vols ${r.st.s1}/${r.st.s2} rho=${r.st.rho.toFixed(2).padStart(5)} T=${r.st.T}  gpu ${r.spot.toFixed(6)}` +
      (r.cf ? `  ${r.cf.name} ${r.cf.v.toFixed(6)} |err| ${err.toExponential(1)}` : "  (painted)") +
      (r.cf && r.cf.kirk !== undefined ? `  Kirk ${r.cf.kirk.toFixed(6)}` : "") +
      `  gpu vs 64-bit ${r.worst.toExponential(1)}  ${r.method}${r.N ? " " + r.N.join("x") : ""} kernel ${r.taps.join("x")} ${r.ms.toFixed(0)} ms`,
  );
  return ok;
};
let fails = 0;
const chk = async (w) => {
  if (!(await row(w))) fails++;
};
const shot = async (name) => SHOTS && panel.screenshot({ path: `${SHOTS}/${name}.png` });
await chk("start");
await shot("ts-exchange-rho05");
for (const rho of [-0.9, 0, 0.9]) {
  await set("tsR", rho);
  await chk(`exchange rho sweep`);
  if (rho !== 0) await shot(`ts-exchange-rho${rho}`);
}
await set("tsR", 0.5);
await page.click('[data-ts="best"]');
await chk("best-of preset");
await set("tsS1", 60);
await set("tsS2", 40);
await set("tsTm", 2);
await chk("best-of wide vols, T=2");
await set("tsK", 120);
await chk("best-of K=120");
await page.click('[data-tsm="direct"]');
await chk("best-of direct");
await page.click('[data-tsm="fft"]');
await set("tsS1", 30);
await set("tsS2", 25);
await set("tsTm", 1);
await page.click('[data-ts="spread"]');
await chk("spread preset");
await set("tsR", 0.9);
await set("tsK", 30);
await chk("spread rho .9 K=30 (Kirk off)");
await shot("ts-spread");
await set("tsR", 0.5);
await set("tsK", 5);
await page.click('[data-ts="paint"]');
await page.click("#tsClear");
// paint a blob and a stripe
const box = await page.locator("#tsPaint").boundingBox();
await page.mouse.move(box.x + box.width * 0.62, box.y + box.height * 0.3);
await page.mouse.down();
for (let t = 0; t <= 20; t++) await page.mouse.move(box.x + box.width * (0.62 + (0.25 * t) / 20), box.y + box.height * (0.3 + (0.4 * t) / 20));
await page.mouse.up();
await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.65);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.6);
await page.mouse.up();
await chk("painted payoff");
await shot("ts-paint");
console.log(fails ? `${fails} FAILED` : "all demo checks PASS");
await browser.close();
await server.close();
process.exit(fails ? 1 : 0);
