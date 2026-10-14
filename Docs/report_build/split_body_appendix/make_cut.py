"""Builds report_cut from report_complete.

report_cut = report_complete with (i) the body (Secs. I-X) copied 1:1, (ii) the appendices reduced to headings, the
"In short" summary of every section and subsection, and the figures and tables, and (iii) no table of contents.

usage: python3 report_complete/split_body_appendix/make_cut.py   (then: cd report_cut && tectonic main.tex)
Run again after report_complete changes (e.g. after merging Marco's frequentist.tex): nothing is edited by hand.
"""
import os, re, shutil

BUILD = os.path.dirname(os.path.abspath(__file__))                      # report_complete/split_body_appendix
SRC = os.path.abspath(os.path.join(BUILD, '..'))                        # report_complete
HERE = os.path.abspath(os.path.join(SRC, '..', 'report_cut'))           # output folder
os.makedirs(HERE + '/sections', exist_ok=True)
os.makedirs(HERE + '/figures', exist_ok=True)

# ---- main.tex without the table of contents
main = open(SRC + '/main.tex').read()
main = re.sub(r'\\setcounter\{tocdepth\}\{\d\}\n', '', main)
main = main.replace('\\tableofcontents\n', '')
open(HERE + '/main.tex', 'w').write(main)
for f in ('preamble.tex', 'biblio.bib'):
    if os.path.exists(SRC + '/' + f):
        shutil.copy(SRC + '/' + f, HERE + '/' + f)
if os.path.exists(SRC + '/mainNotes.bib'):
    shutil.copy(SRC + '/mainNotes.bib', HERE + '/mainNotes.bib')

# ---- body: every input of main.tex except the appendix, copied unchanged
inputs = re.findall(r'\\input\{sections/([^}]+)\}', main)
for n in inputs:
    if n == 'appendix':
        continue
    shutil.copy(f'{SRC}/sections/{n}.tex', f'{HERE}/sections/{n}.tex')

# ---- slimmer versions of the body sections of the cut (cut_overrides/*.tex); the extended report is not changed
OVR = os.path.join(BUILD, 'cut_overrides')
if os.path.isdir(OVR):
    for n in os.listdir(OVR):
        if n.endswith('.tex'):
            shutil.copy(f'{OVR}/{n}', f'{HERE}/sections/{n}')


# ---- appendix: headings + summaries + figures + tables
def expand(t):
    return re.sub(r'\\input\{sections/([^}]+)\}',
                  lambda m: open(f'{SRC}/sections/{m.group(1)}.tex').read() + '\n', t)

app = expand(open(SRC + '/sections/appendix.tex').read())
FL = re.compile(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}', re.S)


def balanced(s, i):
    d = 0
    for k in range(i, len(s)):
        if s[k] == '{': d += 1
        elif s[k] == '}':
            d -= 1
            if d == 0: return k + 1
    return len(s)


def lead(text):
    text = FL.sub('', text)           # a summary never sits inside a float
    j = text.find('\\emph{In short')
    return text[j:balanced(text, j + 5)] if j >= 0 else ''


def header(block):
    """the \\section/\\subsection line followed by its \\label lines"""
    m = re.match(r'(\\(?:section|subsection)\{(?:[^{}]|\{[^{}]*\})*\}\n)((?:\\label\{[^}]*\}\n)*)', block)
    return m.group(1) + m.group(2) if m else block.split('\n')[0] + '\n'


def keep(block):
    h = header(block)
    body = block[len(h):]
    out = h
    ld = lead(body)
    if ld:
        out += ld + '\n'
    out += ''.join(m.group(0).rstrip('\n') + '\n' for m in FL.finditer(body))
    return out


head, _, rest = app.partition('\\section{')
rest = '\\section{' + rest
out = [re.sub(r'\\makeatletter\\addtocontents[^\n]*\n', '', head)]
for sec in re.split(r'(?m)^(?=\\section\{)', rest):
    if not sec.strip():
        continue
    parts = re.split(r'(?m)^(?=\\subsection\{)', sec)
    s = keep(parts[0])
    if not lead(parts[0]) and 'Notation' in parts[0].split('\n')[0]:
        s = header(parts[0]) + '\\emph{In short: the symbols and abbreviations used in the report.}\n' + s[len(header(parts[0])):]
    s += '\n'.join(keep(p) for p in parts[1:])
    out.append(s + '\n\\FloatBarrier\n')
open(HERE + '/sections/appendix.tex', 'w').write('\n'.join(out))

# ---- figures actually used
alltex = ''.join(open(f'{HERE}/sections/{n}').read() for n in os.listdir(HERE + '/sections') if n.endswith('.tex'))
used = set(re.findall(r'includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', alltex))
for u in used:
    src = f'{SRC}/{u}'
    if os.path.exists(src):
        os.makedirs(os.path.dirname(f'{HERE}/{u}'), exist_ok=True)
        shutil.copy(src, f'{HERE}/{u}')
    else:
        print('missing figure', u)
print('report_cut built:', len(used), 'figures,', len(inputs) - 1, 'body files')
