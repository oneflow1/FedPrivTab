const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/data/home/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  await page.goto('http://127.0.0.1:8503', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(5000);
  console.log(await page.locator('body').innerText({timeout: 10000}));
  const buttons = await page.locator('button, [role="radio"], label').evaluateAll(els => els.slice(0,100).map((el, i) => ({i, tag: el.tagName, role: el.getAttribute('role'), text: el.textContent})));
  console.log(JSON.stringify(buttons, null, 2));
  await browser.close();
})();
