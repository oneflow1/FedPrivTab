const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const baseUrl = process.env.FEDPRIVTAB_UI_URL || 'http://127.0.0.1:8501';
const chromePath = '/opt/data/home/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome';
const outDir = '/opt/data/report/FedPrivTab/ui-verification-full';
fs.mkdirSync(outDir, { recursive: true });

async function waitForText(page, pattern, timeout = 90000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const text = await page.locator('body').innerText().catch(() => '');
    if (typeof pattern === 'string' ? text.includes(pattern) : pattern.test(text)) return text;
    await page.waitForTimeout(750);
  }
  throw new Error(`Timed out waiting for ${pattern}`);
}

async function login(page, username, password) {
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await waitForText(page, '登录工作台');
  await page.getByLabel('用户名').fill(username);
  await page.getByLabel('密码').fill(password);
  await page.getByRole('button', { name: '登录' }).click();
  await waitForText(page, /首页|Workspace Status|WORKSPACE STATUS/);
}

async function clickNav(page, name) {
  const candidates = [
    page.getByRole('radio', { name: new RegExp(name) }),
    page.locator('label').filter({ hasText: name }).first(),
    page.getByText(name, { exact: true }).first(),
    page.getByText(new RegExp(name)).first(),
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
  while (Date.now() - start < 60000) {
    const headings = await page.locator('h1, h2, h3').evaluateAll(els => els.map(el => el.textContent || '').join('\n')).catch(() => '');
    if (headings.includes(name)) return;
    await page.waitForTimeout(750);
  }
  throw new Error(`Timed out waiting for heading ${name}`);
}

function pass(name, evidence) {
  return { name, status: 'PASS', evidence };
}

function fail(name, evidence) {
  return { name, status: 'FAIL', evidence };
}

(async () => {
  const browser = await chromium.launch({ executablePath: chromePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  const consoleMessages = [];
  page.on('console', msg => consoleMessages.push(`${msg.type()}: ${msg.text()}`));
  page.on('pageerror', err => consoleMessages.push(`pageerror: ${err.message}`));

  const results = [];
  await login(page, 'researcher', 'research123');
  const researcherHome = await page.locator('body').innerText();
  const managerPages = ['首页', '客户端管理页', '数据上传与审核页', '数据分析页', '实验配置页', '训练监控页', '结果分析页', '报告导出页'];
  const managerMissing = managerPages.filter(text => !researcherHome.includes(text));
  results.push(managerMissing.length === 0 ? pass('研究员可访问全部管理页面', managerPages.join(', ')) : fail('研究员可访问全部管理页面', `缺少: ${managerMissing.join(', ')}`));

  const pageRequirements = {
    '首页': ['客户端', '实验概览', '集中式 MLP', 'FedAvg + MLP', 'DP-FedAvg + MLP'],
    '客户端管理页': ['创建客户端账号', '客户端清单', '启用'],
    '数据上传与审核页': ['一键数据预处理', '上传 CSV 数据', '一键预处理并校验', '审核通过并启用数据', '审核驳回', '数据预览'],
    '数据分析页': ['统计摘要', '标签分布', '客户端标签分布', '相关性热力图'],
    '实验配置页': ['Non-IID', 'alpha', '裁剪阈值 C', '噪声倍率 sigma', '64,32'],
    '训练监控页': ['训练控制', '全部方案', '开始训练', '数据状态'],
    '结果分析页': ['暂无训练结果', '训练监控页'],
    '报告导出页': ['生成 Markdown 报告', '下载 Markdown 报告', 'FedPrivTab 实验报告'],
  };

  const screenshots = {};
  for (const pageName of managerPages) {
    await clickNav(page, pageName);
    const body = await page.locator('body').innerText();
    const screenshotPath = path.join(outDir, `${pageName}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    screenshots[pageName] = screenshotPath;
    const missing = pageRequirements[pageName].filter(text => !body.includes(text));
    results.push(missing.length === 0 ? pass(`${pageName}需求文案覆盖`, '关键区域均存在') : fail(`${pageName}需求文案覆盖`, `缺少: ${missing.join(', ')}`));
  }

  await page.context().clearCookies();
  await login(page, 'client', 'client123');
  const clientHome = await page.locator('body').innerText();
  results.push(!clientHome.includes('客户端管理页') && !clientHome.includes('实验配置页') && !clientHome.includes('训练监控页')
    ? pass('客户端用户隐藏管理/训练入口', '未显示客户端管理、实验配置、训练监控入口')
    : fail('客户端用户隐藏管理/训练入口', '仍显示管理或训练入口'));
  await clickNav(page, '数据上传与审核页');
  const clientUpload = await page.locator('body').innerText();
  results.push(!clientUpload.includes('审核通过并启用数据') || clientUpload.includes('disabled')
    ? pass('客户端用户不能审核数据', '客户端视图未开放审核动作')
    : fail('客户端用户不能审核数据', '客户端视图仍可见审核动作'));

  const summary = { url: baseUrl, timestamp: new Date().toISOString(), consoleMessages, screenshots, results };
  fs.writeFileSync(path.join(outDir, 'ui-full-verification.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  const failures = results.filter(item => item.status === 'FAIL');
  await browser.close();
  if (failures.length) process.exit(1);
})();
