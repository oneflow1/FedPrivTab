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
  const managerPages = ['首页', '客户端管理页', '数据分析页', '数据预处理页', '实验训练页', '结果分析页'];
  results.push(pass('研究员可访问新导航页面', managerPages.join(', ')));
  results.push(!researcherHome.includes('数据状态') ? pass('Workspace Status 精简', '只展示客户端和训练结果') : fail('Workspace Status 精简', '仍展示数据状态'));

  const pageRequirements = {
    '首页': ['客户端', '实验概览', '集中式 MLP', 'FedAvg + MLP', 'DP-FedAvg + MLP'],
    '客户端管理页': ['客户端账号', '客户端账号清单', '修改密码'],
    '数据预处理页': ['文件上传', '上传 CSV 数据', '目标变量', '缺失值摘要和处理方式', '数值标准化', '一键处理并保存版本', '处理版本记录'],
    '数据分析页': ['上传用于数据分析的 CSV 文件', '统计摘要', '字段分布', '选择字段查看分布', '特征均值', '相关性热力图'],
    '实验训练页': ['实验参数配置', '勾选训练方案', '集中式 MLP 数据版本', 'FedAvg / DP-FedAvg 数据版本', '开始训练'],
    '结果分析页': ['结果分析页', '报告导出'],
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
  await login(page, 'client-1', 'client123');
  const clientHome = await page.locator('body').innerText();
  results.push(!clientHome.includes('客户端管理页') && !clientHome.includes('实验训练页')
    ? pass('客户端用户隐藏管理/训练入口', '未显示客户端管理、实验训练入口')
    : fail('客户端用户隐藏管理/训练入口', '仍显示管理或训练入口'));
  await clickNav(page, '数据预处理页');
  const clientUpload = await page.locator('body').innerText();
  results.push(clientUpload.includes('客户端分布式 MLP 数据')
    ? pass('客户端预处理用途正确', '客户端处理数据用于分布式 MLP')
    : fail('客户端预处理用途正确', '未显示客户端分布式处理用途'));

  const summary = { url: baseUrl, timestamp: new Date().toISOString(), consoleMessages, screenshots, results };
  fs.writeFileSync(path.join(outDir, 'ui-full-verification.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  const failures = results.filter(item => item.status === 'FAIL');
  await browser.close();
  if (failures.length) process.exit(1);
})();
