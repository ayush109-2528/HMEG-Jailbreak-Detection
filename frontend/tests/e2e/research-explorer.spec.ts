import {test,expect} from '@playwright/test';
for(const theme of ['dark','light']) test(`Research Explorer interactions in ${theme}`,async({page,context})=>{
 await context.grantPermissions(['clipboard-read','clipboard-write']);
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/research-explorer');await expect(page.locator('.research-node').first()).toBeVisible();
 if(theme==='light')await page.getByRole('button',{name:'Toggle light theme'}).click();
 const count=await page.locator('.paper-card').count();expect(count).toBeGreaterThan(8);
 await page.getByLabel('Search research papers').fill('no-paper-xyz');await expect(page.locator('.paper-card')).toHaveCount(0);
 await page.getByLabel('Search research papers').fill('');await expect(page.locator('.paper-card')).toHaveCount(count);
 await page.locator('.paper-select').filter({hasText:'Embedding-based classifiers'}).click();
 for(const name of ['Overview','Method','Evidence','Connections','Cite']){await page.getByRole('tab',{name,exact:true}).click();await expect(page.locator('.paper-details [role=tabpanel]')).not.toBeEmpty()}
 await page.getByLabel('Citation format').selectOption('BibTeX');await page.getByRole('button',{name:'Copy citation',exact:true}).click();await expect(page.getByText('Copied citation',{exact:true})).toBeVisible();
 await page.getByRole('tab',{name:'Timeline',exact:true}).click();await expect(page.locator('.research-timeline')).toContainText('2024');
 await page.getByRole('tab',{name:'List',exact:true}).click();await expect(page.locator('.research-list-view article')).toHaveCount(count);
 await page.locator('.paper-card input').nth(0).check();await page.locator('.paper-card input').nth(1).check();await page.getByRole('tab',{name:/^Compare/}).click();await expect(page.locator('.research-comparison table')).toBeVisible();
 await page.getByLabel('Include HMEG').check();await expect(page.locator('.research-comparison table')).toContainText('English only');
 await page.getByRole('tab',{name:'Graph',exact:true}).click();await page.getByRole('button',{name:'Return to HMEG'}).click();
 await page.getByRole('button',{name:'Present',exact:true}).click();await expect(page.locator('.research-explorer')).toHaveClass(/presenting/);await page.keyboard.press('ArrowRight');await expect(page.locator('.paper-details h2')).not.toHaveText('HMEG');await page.keyboard.press('Home');await expect(page.locator('.paper-details h2')).toHaveText('HMEG');await page.getByRole('button',{name:'Exit presentation',exact:true}).click();
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(250);expect(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)).toBe(false);expect(errors).toEqual([]);
});
test('licensed PDF navigation and private-source fallback',async({page})=>{
 await page.goto('/research-explorer');await page.locator('.paper-select').filter({hasText:'Embedding-based classifiers'}).click();await page.getByRole('button',{name:'Read PDF',exact:true}).click();await expect(page.getByLabel('PDF page',{exact:true})).toHaveValue('1');await page.getByRole('button',{name:'Next page',exact:true}).click();await expect(page.getByLabel('PDF page',{exact:true})).toHaveValue('2');await page.getByLabel('Search inside PDF').fill('Random Forest');await page.getByRole('button',{name:'Find in PDF'}).click();await expect(page.getByText(/matching pages/)).not.toHaveText('0 matching pages');await page.getByRole('button',{name:'Close PDF reader'}).click();
 await page.locator('.paper-select').filter({hasText:'Llama Guard'}).click();await page.getByRole('button',{name:'Read PDF',exact:true}).click();await expect(page.getByRole('dialog')).toContainText('never uploaded');await expect(page.locator('input[type=file]')).toBeVisible();
});
