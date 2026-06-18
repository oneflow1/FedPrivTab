const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/data/home/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  page.on('console', msg => console.log('CONSOLE', msg.type(), msg.text()));
  page.on('pageerror', err => console.log('PAGEERROR', err.message));
  await page.goto('http://127.0.0.1:8503', { waitUntil: 'domcontentloaded', timeout: 60000 });
  for (let i=0;i<90;i++) {
    await page.waitForTimeout(1000);
    const txt = await page.locator('body').innerText().catch(e => '');
    if (txt.includes('FedPrivTab') || txt.includes('首页') || txt.includes('客户端管理页')) {
      console.log('READY', i, txt.slice(0,1000));
      await page.screenshot({path:'/opt/data/report/FedPrivTab/ui-verification/ready.png', fullPage:true});
      await browser.close();
      return;
    }
    if (i % 10 === 0) console.log('WAIT', i, txt.length, JSON.stringify(txt.slice(0,120)));
  }
  const txt = await page.locator('body').innerText().catch(e => '');
  console.log('NOT_READY', txt.length, JSON.stringify(txt.slice(0,1000)));
  await page.screenshot({path:'/opt/data/report/FedPrivTab/ui-verification/not_ready.png', fullPage:true});
  await browser.close();
  process.exit(2);
})();
