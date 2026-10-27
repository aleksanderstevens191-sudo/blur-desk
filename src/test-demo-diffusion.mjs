// node src/test-demo-diffusion.mjs [shots dir] — panel 16 (DIFFUSION) of the demo in headless
// Chromium on its software GPU, driven like a user: the shared blur slider at five levels,
// where both GPU blurs must stay within 1e-4 of their closed forms (Black's formula on the
// left, Merton's mixture with the blur's variance on the right), then the play button: 65,536
// points of noise run back to the market's odds, which a KS test must not reject (p > 0.01).
// Needs npm run build and playwright, as src/test-gpu.mjs does. With a directory, saves
// screenshots there.
import { preview } from "vite";
import { chromium } from "playwright";

const SHOTS = process.argv[2];
const server = await preview({ logLevel: "warn", preview: { port: 4178, host: "127.0.0.1" } });
const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1180, height: 900 } });
page.on("pageerror", (e) => console.error(`page error: ${e.message}`));
page.on("console", (m) => m.type() === "error" && console.error(`console: ${m.text()}`));
await page.goto(server.resolvedUrls.local[0]);
await page.waitForFunction(() => window.diffusion && window.diffusion.ready && window.diffusion.live, null, { timeout: 600e3, polling: 500 });
const panel = page.locator("#dfPanel");
await panel.scrollIntoViewIfNeeded();
const shot = async (name) => SHOTS && panel.screenshot({ path: `${SHOTS}/${name}.png` });
const settle = () =>
  page.waitForFunction(() => {
    const d = window.diffusion;
    return !d.busy && (d.failure || (d.live && d.live.seq === d.asked));
  }, null, { timeout: 600e3, polling: 200 });
let fails = 0;
const line = (ok, s) => {
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s}`);
};

for (const j of [0, 32, 64, 96, 128]) {
  await page.evaluate((j) => window.diffusion.setLevel(j), j);
  await settle();
  const r = await page.evaluate(() => {
    const d = window.diffusion;
    return d.failure ? { err: d.failure.message } : d.live;
  });
  if (r.err) {
    line(false, `level ${j}: ${r.err}`);
    continue;
  }
  line(r.worst < 1e-4, `blur level ${String(j).padStart(3)}: price vs Black ${r.pErr.toExponential(1)} per $100, density vs Merton + blur ${r.dErr.toExponential(1)} per unit of y`);
  if (j === 64 || j === 128) await shot(`df-level${j}`);
}
for (const pure of [false, true]) {
  await page.evaluate((p) => window.diffusion.setPure(p), pure);
  await page.evaluate(() => window.diffusion.play());
  await page.waitForTimeout(1500);
  if (!pure) await shot("df-playing");
  await page.waitForFunction(() => !window.diffusion.playing && window.diffusion.run, null, { timeout: 600e3, polling: 200 });
  const r = await page.evaluate(() => {
    const x = window.diffusion.run;
    return { D: x.ks.D, p: x.ks.p, pDown: x.pDown, mDown: x.mDown, ms: x.ms };
  });
  const what = pure ? "from pure N(0,1)" : "from the blurred odds";
  const msg = `reverse run ${what}: 65,536 points, KS D ${r.D.toFixed(4)} p ${r.p.toFixed(2)}; P(fall > 10%) ${(100 * r.pDown).toFixed(2)}% vs ${(100 * r.mDown).toFixed(2)}% (${r.ms.toFixed(0)} ms)`;
  if (pure) console.log(`INFO ${msg} (Stage 2: pure noise is 0.0039 from the blurred odds in L1)`);
  else line(r.p > 0.01, msg);
  await shot(pure ? "df-done-pure" : "df-done");
}
console.log(fails ? `\n${fails} failed` : "\npanel 16 passes");
await browser.close();
await server.close();
process.exit(fails ? 1 : 0);
