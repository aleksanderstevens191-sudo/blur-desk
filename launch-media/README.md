# Launch media

What posting Blur Desk on X needs: two 15-second clips, the post text, and alt text.
The clips are recorded on the Mac from the live site; nothing in here is a recording yet.

| File | What it is |
| --- | --- |
| `post.md` | The post and a 3-post thread, with alt text for both clips |
| `record1-first-frame.png` | What `/?record=1` opens on now (headless Chromium, software GPU: the timings in it are not Mac numbers, don't post it) |
| `record2-first-frame.png` | The same for `/?record=2` |

## Both clips open on a filled frame

Each take holds the finished frame for its first second, then starts over, so the
post's first frame and its thumbnail are the result rather than an empty chart. The
same frame is on screen while the page waits for Space, so a take that starts a little
early still opens on it.

- `/?record=1`: 0–1 s the market's odds for Dec 18 (7.6% chance SPY falls more than 10%),
  then draw (1–3.9 s), blur by expiry (4.2–8.6 s), swap in the SPY fit (9 s), odds (12–15 s).
- `/?record=2`: 0–1 s the rebuilt odds and read-off (7.44% rebuilt vs the market's 7.62%:
  the same seeded run the take ends on), then the odds (1 s), the blur sweep (2.2–6.6 s),
  Play (7 s), the read-off (11.4–15 s).

## Recording on the Mac (about 20 minutes)

1. In Chrome, open `blur-desk.vercel.app/?record=1`. Type the address in full and check
   the URL bar still says `?record=1` (autocomplete dropped it on two earlier takes).
2. Wait for `READY · SPACE PLAYS 15 S` under the frame. Click once on the page so it has
   the keyboard. Then leave the pointer on that READY line, below the frame, so it
   stays out of the recording.
3. Cmd + Shift + 5 → Record Selected Portion. Draw the box tightly around the square
   frame (leave out the hint line under it). Click Record.
4. Press Space. The take ends at `DONE` after 15 s. Stop the recording from the menu bar.
5. In QuickTime, Edit → Trim: start on the first frame after Space (the filled frame
   holds 1 s, so there is slack), end on `DONE`.
6. Repeat with `blur-desk.vercel.app/?record=2`. It takes longer to say READY (it builds
   129 noise levels on the GPU first).

X takes the `.mov` files as they are (up to 140 s, 512 MB). If one is too large, or to
match the repo's 1080 × 1080 MP4:

```sh
ffmpeg -i take.mov -vf "scale=1080:1080:flags=lanczos,fps=30" -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart -an blur-desk-15s.mp4
```

## Real phone check (5 minutes)

Open `blur-desk.vercel.app` on the phone, scroll every panel, press Play on panel 16, and
screenshot anything cut off. The phone-width fixes so far were checked only at 390 px in
headless Chromium.
