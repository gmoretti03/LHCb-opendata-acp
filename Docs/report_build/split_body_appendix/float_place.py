"""Rule: a figure or table is placed right after the paragraph that cites it first, and every float is cited in the text.

place_floats(text) moves each float of a section file to the position just after the first paragraph (of the same file,
outside floats, within the same section/subsection) that cites one of its labels; floats that are never cited in the
text are returned so that the caller can report or fix them.
"""
import re

FLOAT = re.compile(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}', re.S)
HEAD = re.compile(r'\\(?:section|subsection|subsubsection)\{')


def _blocks(t):
    out = []
    pos = 0
    for m in FLOAT.finditer(t):
        out.extend(('text', p) for p in re.split(r'(?<=\n)\s*\n', t[pos:m.start()]) if p != '')
        out.append(('float', m.group(0)))
        pos = m.end()
    out.extend(('text', p) for p in re.split(r'(?<=\n)\s*\n', t[pos:]) if p != '')
    return out


def _cites(par, labs):
    for l in labs:
        if re.search(r'\\(?:ref|eqref|autoref)\{' + re.escape(l) + r'\}', par):
            return True
        if re.search(r'\\(?:appref|apponly)\{' + re.escape(l) + r'\}|\\apprefs\{[^}]*' + re.escape(l) + r'|\\secorapp\{[^}]*\}\{' + re.escape(l) + r'\}', par):
            return True
    return False


def place_floats(t):
    """Each float goes right after the first paragraph that cites it in its own subsection; if its own subsection does not
    cite it, after the first citing paragraph of the same section; otherwise it stays where it is."""
    blocks = _blocks(t)
    n = len(blocks)
    grp, sec = [], []
    g = s_ = 0
    for k, b in blocks:
        if k == 'text':
            if re.search(r'\\section\{', b):
                s_ += 1; g += 1
            elif re.search(r'\\(?:subsection|subsubsection)\{', b):
                g += 1
        grp.append(g); sec.append(s_)
    after = {}                                   # index of a text block -> floats to put after it
    stay = []
    for i, (k, b) in enumerate(blocks):
        if k != 'float':
            continue
        labs = re.findall(r'\\label\{([^}]+)\}', b)
        own = [j for j in range(n) if blocks[j][0] == 'text' and grp[j] == grp[i] and _cites(blocks[j][1], labs)]
        sect = [j for j in range(n) if blocks[j][0] == 'text' and sec[j] == sec[i] and _cites(blocks[j][1], labs)]
        tgt = own[0] if own else (sect[0] if sect else None)
        if tgt is None:
            stay.append(i)
        else:
            after.setdefault(tgt, []).append(b)
    out = []
    pending_stay = set(stay)
    last_text = None
    for i, (k, b) in enumerate(blocks):
        if k == 'float':
            if i in pending_stay:
                out.append(('float', b))
            continue
        out.append(('text', b))
        for fb in after.get(i, []):
            out.append(('float', fb))
    res = ''
    for k, b in out:
        res += b if k == 'text' else ('\n' if not res.endswith('\n\n') else '') + b + '\n\n'
    return re.sub(r'\n{3,}', '\n\n', res)


def never_cited(texts):
    """labels of floats not cited by any text paragraph of any of the files"""
    alltext = '\n'.join(FLOAT.sub('', t) for t in texts.values())
    res = []
    for n, t in texts.items():
        for m in FLOAT.finditer(t):
            labs = re.findall(r'\\label\{([^}]+)\}', m.group(0))
            if not _cites(alltext, labs):
                res.append((n, labs))
    return res
