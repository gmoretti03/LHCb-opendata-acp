"""Builds report_short from report_cut: the cut without the appendix and without any citation of it.

The rule is just that:
  (i)   the appendix is not input;
  (ii)  every citation of the appendix is removed from the body: \\appref, \\apprefs, \\appfig, \\appfigs, \\apprefsf,
        \\apponly and \\secorapp, with their arguments (the blank before the macro goes too when a punctuation mark follows);
  (iii) the two things of the appendix that sit in the body source are dropped: the plain pointer to the notation
        appendix of the introduction and the cut-flow table of Appendix A (called from the conclusions of the cut to share
        a page with Fig. XIV);
  (iv)  the few edits that belong to the short only are listed in SHORT_TWEAKS below (figure widths and one sentence);
        nothing else is edited: the text, the floats and the figures are those of the cut.

The text of the cut is edited in cut_final/ (copied over report_cut/sections, see cut_final/README.txt).

usage (from Docs): python3 report_build/split_body_appendix/make_short_from_cut.py
then               cd report_short && tectonic main.tex
This replaces make_short.py (long pipeline with exact-string rewrites and short_overrides, kept for history).
"""
import glob, os, re, shutil, sys

DOCS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CUT = os.path.join(DOCS, 'report_cut')
OUT = os.path.join(DOCS, 'report_short')

MACROS = {'appref': 1, 'apponly': 1, 'apprefs': 2, 'appfig': 2, 'appfigs': 4, 'apprefsf': 3, 'secorapp': 2}
MACRO_RE = re.compile(r'\\(appref|apponly|apprefs|appfigs|appfig|apprefsf|secorapp)(?![A-Za-z])')


def end_of_args(s, i, n):
    """index after n balanced {...} arguments starting at s[i], or None"""
    for _ in range(n):
        while i < len(s) and s[i] in ' \n':
            i += 1
        if i >= len(s) or s[i] != '{':
            return None
        d = 0
        j = i
        while j < len(s):
            if s[j] == '{':
                d += 1
            elif s[j] == '}':
                d -= 1
                if d == 0:
                    break
            j += 1
        i = j + 1
    return i


def strip_citations(s, name):
    n = 0
    while True:
        m = MACRO_RE.search(s)
        if not m:
            return s, n
        e = end_of_args(s, m.end(), MACROS[m.group(1)])
        if e is None:
            sys.exit(f'{name}: malformed {m.group(0)}')
        a = m.start()
        if a > 0 and s[a - 1] == ' ' and e < len(s) and s[e] in '.,;:)':
            a -= 1
        s = s[:a] + s[e:]
        n += 1


def drop_appendix_items(s, name):
    # plain pointer to the notation appendix (introduction)
    ptr = '; the symbols are listed in Appendix~\\ref{sec:notation}.'
    if ptr in s:
        s = s.replace(ptr, '.', 1)
    # cut-flow table of Appendix A, with the comment line above it
    if '\\label{tab:cutflow}' in s:
        k = s.index('\\label{tab:cutflow}')
        i = max(s.rfind('\\begin{figure*}', 0, k), s.rfind('\\begin{figure}', 0, k))
        j = s.index('\\end{figure', k)
        j = s.index('\n', j) + 1
        c = s.rfind('\n%', 0, i)
        if c != -1 and s[c + 1:i].strip().startswith('%') and 'cut-flow' in s[c:i]:
            i = c + 1
        s = s[:i].rstrip('\n') + '\n' + s[j:]
    return s


# edits that belong to the short only: (section, text of the cut, text of the short); each must be found exactly once
SHORT_TWEAKS = [
    ('ml', '\\includegraphics[width=\\columnwidth]{figures/short_ml_three.pdf}',
           '\\includegraphics[width=0.9\\columnwidth]{figures/short_ml_three.pdf}'),
    ('systematics', '\\includegraphics[width=0.80\\textwidth]{figures/short_pid_scan.pdf}',
                    '\\includegraphics[width=0.75\\textwidth]{figures/short_pid_scan.pdf}'),
    ('conclusions', 'its $68\\%$ is a property of the procedure that the bootstrap and the toy experiments verified.',
                    'its $68\\%$ is verified by the bootstrap and the toy experiments.'),
]

main = open(os.path.join(CUT, 'main.tex')).read()
main = main.replace('\\input{sections/appendix}\n\n', '').replace('\\input{sections/appendix}\n', '')
assert 'sections/appendix' not in main
os.makedirs(os.path.join(OUT, 'sections'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
for f in glob.glob(os.path.join(OUT, 'sections', '*.tex')):
    os.remove(f)                                              # no section of an earlier build survives
open(os.path.join(OUT, 'main.tex'), 'w').write(main)
for f in ('preamble.tex', 'biblio.bib', 'mainNotes.bib'):
    if os.path.exists(os.path.join(CUT, f)):
        shutil.copy(os.path.join(CUT, f), os.path.join(OUT, f))

total = 0
for name in re.findall(r'\\input\{sections/([^}]+)\}', main):
    s = open(os.path.join(CUT, 'sections', name + '.tex')).read()
    s, n = strip_citations(s, name)
    s = drop_appendix_items(s, name)
    for sec, a, b in SHORT_TWEAKS:
        if sec == name:
            assert s.count(a) == 1, f'{name}: tweak not found: {a[:50]}'
            s = s.replace(a, b, 1)
    assert 'Appendix' not in s and 'tab:cutflow' not in s, f'{name}: appendix left in the text'
    open(os.path.join(OUT, 'sections', name + '.tex'), 'w').write(s)
    total += n
    # the figures of the body that the short needs and does not have yet
    for fig in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s):
        if not os.path.exists(os.path.join(OUT, fig)):
            shutil.copy(os.path.join(CUT, fig), os.path.join(OUT, fig))
print(f'report_short: the cut without the appendix and without its {total} citations')
