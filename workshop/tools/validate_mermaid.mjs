// Usage from repository root after audit_materials.py:
// node workshop/tools/validate_mermaid.mjs /tmp/ragworkshop-mermaid/node_modules
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire(path.join(path.resolve(process.argv[2]), 'validation.cjs'));
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const { default: mermaid } = await import(pathToFileURL(require.resolve('mermaid')));
mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });
const diagrams = JSON.parse(fs.readFileSync('/tmp/ragworkshop-diagrams.json', 'utf8'));
for (const diagram of diagrams) {
    await mermaid.parse(diagram.code);
    process.stdout.write(`Parsed ${diagram.file}\n`);
}
process.stdout.write(`${diagrams.length} Mermaid diagrams parsed successfully\n`);
