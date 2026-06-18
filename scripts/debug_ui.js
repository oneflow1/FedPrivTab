const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/data/home/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  page.on('console', msg => console.log('CONSOLE', msg.type(), msg.text()));
  page.on('pageerror', err => console.log('PAGEERROR', err.message));
  page.on('requestfailed', req => console.log('REQFAILED', req.url(), req.failure()?.errorText));
  await page.goto('http://127.0.0.1:8503', { waitUntil: 'load', timeout: 60000 });
  for (let i=0;i<15;i++) {
    await page.waitForTimeout(1000);
    const txt = await page.locator('body').innerText().catch(e => 'ERR '+e.message);
    console.log('TICK', i, 'len', txt.length, JSON.stringify(txt.slice(0,200)));
    const html = await page.locator('body').innerHTML().catch(e => 'ERR '+e.message);
    console.log('HTML', i, html.slice(0,200).replace(/\n/g,' '));
  }
  await page.screenshot({path:'/opt/data/report/FedPrivTab/ui-verification/debug.png', fullPage:true});
  await browser.close();
})();
