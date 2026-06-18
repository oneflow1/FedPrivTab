const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const baseUrl = 'http://127.0.0.1:8503';
const chromePath = '/opt/data/home/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome';
const outDir = '/opt/data/report/FedPrivTab/ui-verification';
fs.mkdirSync(outDir, { recursive: true });

const pages = [
  '首页',
  '客户端管理页',
  '数据上传与审核页',
  '数据分析页',
  '实验配置页',
  '训练监控页',
  '结果分析页',
  '报告导出页',
];

function issue(number, title, status, evidence) {
  return { number, title, status, evidence };
}

(async () => {
  const browser = await chromium.launch({ executablePath: chromePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  const consoleMessages = [];
  page.on('console', msg => consoleMessages.push(`${msg.type()}: ${msg.text()}`));
  page.on('pageerror', err => consoleMessages.push(`pageerror: ${err.message}`));
  await page.goto(baseUrl, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(3000);

  const results = [];
  const screenshots = {};

  for (const pageName of pages) {
    const radio = page.getByLabel(pageName);
    await radio.click({ timeout: 15000 });
    await page.waitForTimeout(1200);
    const safeName = pageName.replace(/\s+/g, '_');
    const screenshotPath = path.join(outDir, `${safeName}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    screenshots[pageName] = screenshotPath;
  }

  const bodyText = await page.locator('body').innerText();
  const htmlLeak = bodyText.includes('<div class=') || bodyText.includes('fed-sidebar-metric') || bodyText.includes('</div>');
  results.push(issue(4, 'Workspace Status 底部展示异常，HTML 原文被渲染出来', htmlLeak ? 'FAIL' : 'PASS', htmlLeak ? '页面文本仍包含 HTML 原文片段' : '未发现 HTML 原文泄露'));

  for (const [pageName, issueNo] of [
    ['客户端管理页', 5],
    ['数据上传与审核页', 6],
    ['数据分析页', 7],
    ['实验配置页', 8],
    ['训练监控页', 9],
    ['结果分析页', 10],
    ['报告导出页', 11],
  ]) {
    await page.getByLabel(pageName).click({ timeout: 15000 });
    await page.waitForTimeout(800);
    const text = await page.locator('body').innerText();
    let pass = false;
    let evidence = '';
    if (pageName === '客户端管理页') {
      pass = text.includes('新增客户端') && text.includes('状态表') && text.includes('启用');
      evidence = pass ? '存在新增客户端、状态表、启用/禁用控件' : '缺少客户端管理核心控件';
    } else if (pageName === '数据上传与审核页') {
      pass = text.includes('上传 CSV 数据') && text.includes('生成示例数据') && text.includes('执行数据校验') && text.includes('数据预览');
      evidence = pass ? '存在上传、示例生成、校验、预览流程' : '缺少上传审核流程控件';
    } else if (pageName === '数据分析页') {
      pass = text.includes('统计描述') && text.includes('标签分布') && text.includes('相关性热力图');
      evidence = pass ? '存在统计描述、标签分布、相关性热力图' : '缺少分析图表区域';
    } else if (pageName === '实验配置页') {
      pass = text.includes('MLP') && text.includes('IID / Non-IID') && text.includes('FedAvg') && text.includes('差分隐私') && text.includes('当前配置');
      evidence = pass ? '存在 MLP、Non-IID、FedAvg、DP 配置分组和配置摘要' : '缺少配置分组';
    } else if (pageName === '训练监控页') {
      pass = text.includes('训练方案') && text.includes('开始训练') && (text.includes('数据未通过校验') || text.includes('Loss 曲线'));
      evidence = pass ? '存在训练方案选择、开始训练按钮、校验提示/Loss 区域' : '缺少训练控制和状态提示';
    } else if (pageName === '结果分析页') {
      pass = text.includes('暂无训练结果') || text.includes('三方案对比表');
      evidence = pass ? '存在空状态或结果对比表区域' : '缺少结果分析区域';
    } else if (pageName === '报告导出页') {
      pass = text.includes('生成 Markdown 报告') && text.includes('Markdown 报告内容') && text.includes('下载 Markdown 报告');
      evidence = pass ? '存在生成、预览、下载区域' : '缺少报告生成/预览/下载流程';
    }
    results.push(issue(issueNo, `${pageName}布局问题`, pass ? 'PASS' : 'FAIL', evidence));
  }

  // Issue #3 was about a previous login header; current app should not show large login state.
  await page.getByLabel('数据分析页').click({ timeout: 15000 });
  await page.waitForTimeout(800);
  const analysisText = await page.locator('body').innerText();
  const loginHeaderPresent = analysisText.includes('已登录: admin') || analysisText.includes('退出登录');
  results.push(issue(3, '登录状态和退出按钮展示过大且未规整布局到右上角', loginHeaderPresent ? 'FAIL' : 'PASS', loginHeaderPresent ? '仍显示旧登录状态/退出按钮' : '未发现旧版大型登录状态区'));

  const summary = {
    url: baseUrl,
    timestamp: new Date().toISOString(),
    screenshots,
    consoleMessages,
    results,
  };
  fs.writeFileSync(path.join(outDir, 'ui-verification.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  await browser.close();
})();
