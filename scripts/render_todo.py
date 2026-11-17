#!/usr/bin/env python3
"""Deterministic, dependency-free HTML view of TODO.md's small Markdown subset."""
import argparse
from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    # Escape all source text; only our explicit formatting becomes HTML.
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)


def render(source):
    blocks = []
    pending = []
    kind = None
    in_list = False

    def flush():
        nonlocal in_list, kind
        if not pending:
            return
        if kind == 'li':
            if not in_list:
                blocks.append('<ul>')
                in_list = True
        elif in_list:
            blocks.append('</ul>')
            in_list = False
        blocks.append(f'<{kind}>{inline(" ".join(pending))}</{kind}>')
        pending.clear()

    for line in source.splitlines():
        if not line.strip():
            flush()
        elif line.startswith(('# ', '## ', '- ')):
            flush()
            if line.startswith('# '):
                kind, value = 'h1', line[2:]
            elif line.startswith('## '):
                kind, value = 'h2', line[3:]
            else:
                kind, value = 'li', line[2:]
            pending.append(value)
        else:
            if not pending:
                kind = 'p'
            pending.append(line.strip())
    flush()
    if in_list:
        blocks.append('</ul>')
    body = '\n'.join(blocks)
    return f'''<!doctype html>
<!-- Generated from TODO.md by scripts/render_todo.py. Do not edit this view. -->
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KmzSuite living to-do</title>
<style>
:root{{color-scheme:light dark}}body{{max-width:850px;margin:48px auto;padding:0 24px;font:17px/1.65 system-ui,sans-serif}}
h1{{font-size:32px;line-height:1.2}}h2{{font-size:22px;margin-top:32px}}li{{margin:18px 0}}code{{font-size:.85em;overflow-wrap:anywhere}}
nav,footer{{font-size:14px;opacity:.8}}a{{color:inherit}}footer{{margin-top:40px;border-top:1px solid #8885;padding-top:16px}}
</style></head><body>
<nav><a href="../README.md">Project README</a> · <a href="../TODO.md">Canonical Markdown</a></nav>
<main>
{body}
</main><footer>Generated view · Update TODO.md and regenerate · Outstanding work only</footer>
</body></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if the HTML view is stale')
    args = parser.parse_args()
    source, target = ROOT / 'TODO.md', ROOT / 'docs/TODO.html'
    if args.check and not source.exists() and not target.exists():
        # The exported public tree carries no backlog.
        print('No backlog in this checkout; nothing to check.')
        return
    expected = render(source.read_text(encoding='utf-8'))
    if args.check:
        if not target.is_file() or target.read_text(encoding='utf-8') != expected:
            raise SystemExit('docs/TODO.html is stale; run .venv/bin/python scripts/render_todo.py')
        print('TODO Markdown/HTML synchronized.')
    else:
        target.write_text(expected, encoding='utf-8')
        print('Generated docs/TODO.html')


if __name__ == '__main__':
    main()
