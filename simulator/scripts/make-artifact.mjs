/* Build tool: turn dist-artifact/index.html into an artifact-ready fragment:
   - inline /fonts/*.woff2 as data URIs
   - strip doctype/html/head/body wrappers (artifact host supplies them),
     keeping title + style + module script in order */
import { readFileSync, writeFileSync } from 'node:fs';

let html = readFileSync('dist-artifact/index.html', 'utf8');
html = html.replace(/url\((['"]?)\/fonts\/([^'")]+)\1\)/g, (_, __, file) => {
  const b64 = readFileSync(`public/fonts/${file}`).toString('base64');
  return `url(data:font/woff2;base64,${b64})`;
});
const pick = (re) => [...html.matchAll(re)].map((m) => m[0]).join('\n');
const title = `<meta charset="utf-8">\n<title>The Auction — Volume Profile Trainer</title>`;
const styles = pick(/<style[\s\S]*?<\/style>/g);
const scripts = pick(/<script type="module"[\s\S]*?<\/script>/g);
const bodyInner = html.match(/<body[^>]*>([\s\S]*?)<\/body>/)?.[1] ?? '<div id="app"></div>';
writeFileSync('dist-artifact/artifact.html', `${title}\n${styles}\n${bodyInner}\n${scripts}\n`);
console.log('artifact.html written,', Math.round((title + styles + bodyInner + scripts).length / 1024), 'KB');
