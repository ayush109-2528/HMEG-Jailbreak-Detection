import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import {readFileSync} from 'node:fs';
const results=JSON.parse(readFileSync(new URL('../fixtures/routing-results.json',import.meta.url),'utf8'));
for(const layer of [1,2] as const) for(const theme of ['dark','light']) test(`live diagram layer ${layer} in ${theme}`,async({page})=>{
 await page.route('**/api/predict/hmeg',route=>route.fulfill({json:results[String(layer) as '1'|'2']}));
 await page.goto('/live');
 if(theme==='light')await page.getByRole('button',{name:'Toggle light theme',exact:true}).click();
 await page.getByLabel('Prompt to analyze').fill('Recorded execution fixture for diagram regression');
 await page.getByRole('button',{name:'Analyze prompt',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Execution trace',exact:true})).toBeVisible();
 await page.waitForTimeout(5500);
 const flow=page.locator('.flow');await flow.scrollIntoViewIfNeeded();await page.waitForTimeout(300);
 const outside=await flow.evaluate(el=>{
  const bounds=el.getBoundingClientRect();
  return [...el.querySelectorAll('.react-flow__edge-path,.react-flow__edge-text')].filter(n=>{
   const r=n.getBoundingClientRect();return r.left<bounds.left-1||r.right>bounds.right+1||r.top<bounds.top-1||r.bottom>bounds.bottom+1;
  }).map(n=>n.getAttribute('class'));
 });expect(outside).toEqual([]);
 if(layer===1) await expect(page.locator('[data-id="meta"]')).toContainText('skipped');
 const audit=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa']).analyze();
 expect(audit.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))).toEqual([]);
 await flow.screenshot({path:`test-output/checked-live-layer${layer}-${theme}.png`});
});
