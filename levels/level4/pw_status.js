// Playwright snippet (browser_run_code_unsafe filename=levels/level4/pw_status.js):
// reads our tournament row + recent matches from the Level 4 page.
async (page) => {
  const f = page.frames().find(fr => fr.url().includes('colony'));
  const t = await f.evaluate(() => document.body.innerText);
  const lb = t.indexOf('TOURNAMENT :: LEADERBOARD');
  const me = t.indexOf('YOU', lb);
  const tel = t.indexOf('TOURNAMENT :: TELEMETRY');
  const mt = t.indexOf('PLAYED MATCHES');
  return {
    telemetry: t.slice(tel, tel + 260).replace(/\n+/g, ' | '),
    top: t.slice(lb, lb + 420).replace(/\n+/g, ' | '),
    you: t.slice(Math.max(lb, me - 60), me + 60).replace(/\n+/g, ' | '),
    matches: t.slice(mt, mt + 900).replace(/\n+/g, ' | '),
  };
}
