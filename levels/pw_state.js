async (page) => {
  const f = page.frames().find(fr => fr.url().includes('colony'));
  return await f.evaluate(() => {
    const pcs = {};
    document.querySelectorAll('.colony-viz__cell').forEach(c => {
      const m = c.querySelector('.colony-viz__marker');
      if (m && m.getAttribute('aria-label')) {
        const t = m.getAttribute('aria-label')[0];
        pcs[c.getAttribute('aria-label').slice(7)] = m.className.includes('team-cyan') ? t : t.toLowerCase();
      }
    });
    const txt = document.body.innerText;
    const i = txt.indexOf('STATUS');
    return { pcs, status: txt.slice(i, i + 300) };
  });
}
