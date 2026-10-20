import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import {readFileSync} from 'node:fs';
const fixture=JSON.parse(readFileSync(new URL('../fixtures/routing-results.json',import.meta.url),'utf8'))['2'];
for(const theme of ['dark','light']) test(`short input abstention and hover contrast: ${theme}`,async({page})=>{
 await page.route('**/api/predict/hmeg',route=>route.fulfill({json:{...fixture,abstained:true,prediction:'insufficient_context',label:null,final_probability:null,action:'Request more context / manual review',decision_note:'Provide a complete prompt.',raw_model:{prediction:'jailbreak',final_probability:.97}}}));
 await page.goto('/live');if(theme==='light')await page.getByRole('button',{name:'Toggle light theme',exact:true}).click();
 await page.getByLabel('Prompt to analyze').fill('apple');
 const analyze=page.getByRole('button',{name:'Analyze prompt',exact:true});await analyze.hover();await page.waitForTimeout(500);
 expect((await new AxeBuilder({page}).withRules(['color-contrast']).analyze()).violations).toEqual([]);
 await analyze.click();await expect(page.getByRole('heading',{name:'Insufficient context',exact:true})).toBeVisible();
 await expect(page.getByRole('heading',{name:'Jailbreak detected',exact:true})).toHaveCount(0);
 await expect(page.getByRole('heading',{name:'Benign prompt',exact:true})).toHaveCount(0);
 await page.waitForTimeout(5500);
 await page.locator('.recharts-bar-rectangle').first().hover();await page.waitForTimeout(500);
 expect((await new AxeBuilder({page}).withRules(['color-contrast']).analyze()).violations).toEqual([]);
 await page.screenshot({path:`test-output/short-input-${theme}.png`,fullPage:true});
});
