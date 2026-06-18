const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const baseUrl = 'http://127.0.0.1:8503';
const chromePath = '/opt/data/home/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome';
const outDir = '/opt/data/report/FedPrivTab/ui-verification';
fs.mkdirSync(outDir, { recursive: true });

async function waitForText(page, pattern, timeout = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const text = await page.locator('body').innerText().catch(() => '');
    if (typeof pattern === 'string' ? text.includes(pattern) : pattern.test(text)) return text;
    await page.waitForTimeout(1000);
  }
  throw new Error(`Timed out waiting for ${pattern}`);
}

async function mainTitle(page) {
  return await page.locator('h1, h2, h3').evaluateAll(els => els.map(e => e.textContent || '').join('\n')).catch(() => '');
}

async function clickNav(page, name) {
  const candidates = [
    page.getByRole('radio', { name }),
    page.locator('label').filter({ hasText: name }).first(),
    page.getByText(name, { exact: true }).first(),
  ];
  let lastError;
  for (const locator of candidates) {
    try {
      await locator.click({ timeout: 5000 });
      lastError = null;
      break;
    } catch (error) {
      lastError = error;
    }
  }
  if (lastError) throw lastError;
  const start = Date.now();
  while (Date.now() - start < 30000) {
    await page.waitForTimeout(1000);
    const titles = await mainTitle(page);
    const body = await page.locator('body').innerText().catch(() => '');
    if (titles.includes(name) || body.split('\n').slice(0, 120).some(line => line.trim() === name)) return;
  }
  await page.waitForTimeout(3000);
}

function evaluateIssue(number, title, pass, evidence, screenshot) {
  return { number, title, status: pass ? 'PASS' : 'FAIL', evidence, screenshot };
}

(async () => {
  const browser = await chromium.launch({ executablePath: chromePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1400 } });
  const consoleMessages = [];
  page.on('console', msg => consoleMessages.push(`${msg.type()}: ${msg.text()}`));
  page.on('pageerror', err => consoleMessages.push(`pageerror: ${err.message}`));
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await waitForText(page, '登录工作台', 140000);

  const bodyAtStart = await page.locator('body').innerText();
  if (bodyAtStart.includes('登录工作台')) {
    await page.getByLabel('用户名').fill('admin');
    await page.getByLabel('密码').fill('admin123');
    await page.getByRole('button', { name: '登录' }).click();
  }
  await waitForText(page, /首页|Workspace Status|WORKSPACE STATUS/, 60000);

  const pages = ['首页', '客户端管理页', '数据上传与审核页', '数据分析页', '实验配置页', '训练监控页', '结果分析页', '报告导出页'];
  const pageTexts = {};
  const screenshots = {};
  const titles = {};
  for (const name of pages) {
    await clickNav(page, name);
    await page.waitForTimeout(2500);
    const body = await page.locator('body').innerText();
    pageTexts[name] = body;
    titles[name] = await mainTitle(page);
    const screenshotPath = path.join(outDir, `${name}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    screenshots[name] = screenshotPath;
  }

  const allText = Object.values(pageTexts).join('\n');
  const results = [];
  const htmlLeak = allText.includes('<div class=') || allText.includes('fed-sidebar-metric') || allText.includes('</div>');
  results.push(evaluateIssue(4, 'Workspace Status 底部展示异常，HTML 原文被渲染出来', !htmlLeak, htmlLeak ? '仍发现 HTML 原文泄露' : '所有页面文本中未发现 HTML 原文/样式片段泄露', screenshots['首页']));

  const loginTooLarge = allText.includes('已登录: admin') || allText.includes('退出登录');
  const userAreaCompact = allText.includes('admin') && allText.includes('系统管理员');
  results.push(evaluateIssue(3, '登录状态和退出按钮展示过大且未规整布局到右上角', !loginTooLarge && userAreaCompact, loginTooLarge ? '仍存在旧版“已登录/退出登录”大块区域' : '未发现旧版登录大提示和退出按钮；用户信息以紧凑方式展示', screenshots['数据分析页']));

  const checks = [
    [5, '客户端管理页布局不够简单美观', '客户端管理页', ['添加客户端', '客户端清单', '启用']],
    [6, '数据上传与审核页布局不够简单美观', '数据上传与审核页', ['上传', '示例数据', '执行数据校验', '数据预览']],
    [7, '数据分析页布局不够简单美观', '数据分析页', ['统计摘要', '标签分布', '特征均值', '相关性热力图']],
    [8, '实验配置页布局不够简单美观', '实验配置页', ['模型结构', '数据分布', '联邦训练', '差分隐私', '配置摘要']],
    [9, '训练监控页布局不够简单美观', '训练监控页', ['训练控制', '训练方案', '开始训练', '数据状态']],
    [10, '结果分析页布局不够简单美观', '结果分析页', ['暂无训练结果', '前往训练监控页']],
    [11, '报告导出页布局不够简单美观', '报告导出页', ['生成 Markdown 报告', '报告内容', '下载 Markdown 报告']],
  ];

  for (const [number, title, pageName, required] of checks) {
    const body = pageTexts[pageName] || '';
    const missing = required.filter(item => !body.includes(item));
    const correctPage = (titles[pageName] || '').includes(pageName) || body.includes(pageName);
    results.push(evaluateIssue(number, title, correctPage && missing.length === 0, !correctPage ? `切页未到达目标页面；标题: ${titles[pageName]}` : (missing.length ? `缺少关键文案/区域: ${missing.join(', ')}` : `页面包含关键区域: ${required.join(', ')}`), screenshots[pageName]));
  }

  const summary = { url: baseUrl, timestamp: new Date().toISOString(), consoleMessages, titles, results, screenshots };
  fs.writeFileSync(path.join(outDir, 'ui-verification.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  await browser.close();
})();
