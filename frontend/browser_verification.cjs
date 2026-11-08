/**
 * Automated Real-Browser Verification Script for NL2SQL Application.
 *
 * Runs Google Chrome via playwright-core against the live FastAPI backend (8000)
 * and Vite frontend (5173).
 */

const { chromium } = require('playwright-core');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const SCREENSHOT_DIR = 'C:\\Users\\Login\\.gemini\\antigravity\\brain\\d0e9ddb5-b365-4ca9-885d-3d5c4da75cf2\\screenshots';
const FRONTEND_URL = 'http://127.0.0.1:5173';
const BACKEND_URL = 'http://127.0.0.1:8000';

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForUrl(url, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      // Keep waiting
    }
    await sleep(500);
  }
  return false;
}

async function runVerification() {
  console.log('='.repeat(70));
  console.log('  Milestone 5.4 — Browser-Based End-to-End Verification');
  console.log('='.repeat(70));

  // 1. Start FastAPI Backend
  console.log('\n[1/7] Launching FastAPI Backend (port 8000)...');
  const backendProc = spawn(
    'python',
    ['run_server.py', '--port', '8000', '--host', '127.0.0.1'],
    {
      cwd: path.resolve(__dirname, '..'),
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );

  backendProc.stderr.on('data', (d) => {
    // console.log(`[Backend stderr]: ${d}`);
  });

  const backendReady = await waitForUrl(`${BACKEND_URL}/api/health`, 15000);
  if (!backendReady) {
    throw new Error('Backend failed to start within timeout.');
  }
  console.log('  FastAPI Backend is online and responding to /api/health.');

  // 2. Start Vite Development Server
  console.log('\n[2/7] Launching Vite Frontend (port 5173)...');
  const frontendProc = spawn(
    'cmd.exe',
    ['/c', 'npm', 'run', 'dev', '--', '--port', '5173', '--host', '127.0.0.1'],
    {
      cwd: __dirname,
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );

  const frontendReady = await waitForUrl(FRONTEND_URL, 15000);
  if (!frontendReady) {
    throw new Error('Frontend failed to start within timeout.');
  }
  console.log('  Vite Frontend is online at http://127.0.0.1:5173.');

  let browser;
  const consoleLogs = [];
  const pageErrors = [];

  try {
    // 3. Launch Chrome
    console.log('\n[3/7] Launching Google Chrome via Playwright...');
    browser = await chromium.launch({
      executablePath: CHROME_PATH,
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
    });

    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
    });

    const page = await context.newPage();

    page.on('console', (msg) => {
      consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', (err) => {
      pageErrors.push(err.toString());
      console.error('  Browser Page Error:', err);
    });

    // 4. Initial Navigation & Verification
    console.log('\n[4/7] Navigating to Frontend & Verifying Initial State (1440x900)...');
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });

    // Wait for health & schema to populate
    await page.waitForSelector('[data-testid="status-indicator"]', { timeout: 8000 });
    await page.waitForSelector('[data-testid="schema-explorer"]', { timeout: 8000 });

    const title = await page.title();
    console.log(`  Page Title: "${title}"`);

    const headerText = await page.innerText('[data-testid="app-header"]');
    console.log('  Header Content:', headerText.replace(/\n+/g, ' | '));

    // Capture Screenshot 1: Initial Application
    const sc1Path = path.join(SCREENSHOT_DIR, '01_initial_application.png');
    await page.screenshot({ path: sc1Path, fullPage: true });
    console.log(`  Saved Screenshot 1: ${sc1Path}`);

    // Verify Schema Explorer
    console.log('\n[5/7] Verifying Schema Explorer & Table Expansion...');
    const tableCards = await page.$$('.table-card');
    console.log(`  Tables found in Schema Explorer: ${tableCards.length}`);

    // Expand all tables
    for (const card of tableCards) {
      const headerBtn = await card.$('.table-header-btn');
      const isExpanded = await headerBtn.getAttribute('aria-expanded');
      if (isExpanded === 'false') {
        await headerBtn.click();
        await sleep(200);
      }
    }

    // Capture Screenshot 2: Expanded Schema Explorer
    const sc2Path = path.join(SCREENSHOT_DIR, '02_expanded_schema.png');
    await page.screenshot({ path: sc2Path, fullPage: true });
    console.log(`  Saved Screenshot 2: ${sc2Path}`);

    // 5. Test Natural Language Query & Generation
    console.log('\n[6/7] Testing Query Generation: "Show students with CGPA above 8"...');
    const inputSelector = '[data-testid="question-input"]';
    await page.fill(inputSelector, 'Show students with CGPA above 8');

    const generateBtn = await page.$('[data-testid="generate-button"]');
    await generateBtn.click();

    // Wait for SQL preview to appear
    await page.waitForSelector('[data-testid="sql-preview-section"]', { timeout: 10000 });
    const generatedSql = await page.innerText('[data-testid="generated-sql-code"]');
    console.log(`  Generated SQL in Browser: "${generatedSql.trim()}"`);

    const valStatus = await page.innerText('[data-testid="validation-status-badge"]');
    console.log(`  Validation Badge: "${valStatus.trim()}"`);

    // Verify Copy SQL button
    const copyBtn = await page.$('[data-testid="copy-sql-button"]');
    if (copyBtn) {
      await copyBtn.click();
      await sleep(300);
      const copyText = await copyBtn.innerText();
      console.log(`  Copy Button Feedback: "${copyText.trim()}"`);
    }

    // Capture Screenshot 3: Generated SQL Preview
    const sc3Path = path.join(SCREENSHOT_DIR, '03_generated_sql_preview.png');
    await page.screenshot({ path: sc3Path, fullPage: true });
    console.log(`  Saved Screenshot 3: ${sc3Path}`);

    // 6. Test Explicit Execution Approval & MySQL Results
    console.log('\n[7/7] Testing Execution Approval Gate on Live MySQL...');
    const executeBtn = await page.$('[data-testid="execute-sql-button"]');
    const isExecuteEnabled = await executeBtn.isEnabled();
    console.log(`  Execution Button Enabled: ${isExecuteEnabled}`);

    await executeBtn.click();

    // Wait for results section to render
    await page.waitForSelector('[data-testid="results-section"]', { timeout: 10000 });

    const execStatus = await page.innerText('[data-testid="execution-status-badge"]');
    console.log(`  Execution Status: "${execStatus.trim()}"`);

    const resultRows = await page.$$('[data-testid^="result-row-"]');
    console.log(`  Rows Rendered in Table: ${resultRows.length}`);

    // Verify column headers
    const ths = await page.$$eval('[data-testid="results-data-table"] th', (els) =>
      els.map((e) => e.innerText.trim())
    );
    console.log(`  Table Columns: [${ths.join(', ')}]`);

    // Capture Screenshot 4: Query Results
    const sc4Path = path.join(SCREENSHOT_DIR, '04_query_results.png');
    await page.screenshot({ path: sc4Path, fullPage: true });
    console.log(`  Saved Screenshot 4: ${sc4Path}`);

    // 7. Responsive Viewports Testing
    console.log('\nTesting Responsive Viewports...');
    const viewports = [
      { name: 'Desktop (1440x900)', width: 1440, height: 900 },
      { name: 'Laptop (1280x800)', width: 1280, height: 800 },
      { name: 'Tablet (768x1024)', width: 768, height: 1024 },
      { name: 'Mobile (390x844)', width: 390, height: 844 },
    ];

    for (const vp of viewports) {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await sleep(300);

      // Check for horizontal overflow
      const hasHorizontalScroll = await page.evaluate(() => {
        return document.documentElement.scrollWidth > window.innerWidth;
      });

      console.log(`  ${vp.name}: Horizontal Overflow = ${hasHorizontalScroll}`);

      if (vp.width === 390) {
        // Capture Screenshot 5: Mobile Layout
        const sc5Path = path.join(SCREENSHOT_DIR, '05_mobile_layout.png');
        await page.screenshot({ path: sc5Path, fullPage: true });
        console.log(`  Saved Screenshot 5: ${sc5Path}`);
      }
    }

    console.log('\n' + '='.repeat(70));
    console.log('  Browser Console Messages Captured:', consoleLogs.length);
    console.log('  Page Errors Encountered:', pageErrors.length);
    if (pageErrors.length > 0) {
      console.log('  Errors:', pageErrors);
    }
    console.log('='.repeat(70));

    return {
      success: true,
      consoleLogs,
      pageErrors,
      generatedSql,
      rowCount: resultRows.length,
      columns: ths,
    };
  } finally {
    if (browser) {
      await browser.close();
    }
    console.log('Terminating backend and frontend processes...');
    backendProc.kill();
    frontendProc.kill();
  }
}

runVerification()
  .then((res) => {
    console.log('\nLIVE BROWSER VERIFICATION COMPLETED SUCCESSFULLY!');
    process.exit(0);
  })
  .catch((err) => {
    console.error('\nLIVE BROWSER VERIFICATION FAILED:', err);
    process.exit(1);
  });
