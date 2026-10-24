// node src/test-odds.mjs — the market's odds (odds.js) in 64-bit on the CPU: the
// distribution edge-detected from the SPY fit's blurred prices sums to 1, its mean
// is the parity forward, and its tail odds match Merton's closed-form CDF.
// The GPU (32-bit) half is /check.html?only=odds.
import { readFileSync } from "node:fs";
import { oddsChecks } from "./odds-check.js";

const SPY = JSON.parse(readFileSync(new URL("./spy_fit.json", import.meta.url)));
const fails = await oddsChecks(console.log, SPY, { gpu: false });
console.log(fails ? `\n${fails} failed` : "\nall market-odds checks pass");
process.exit(fails ? 1 : 0);
