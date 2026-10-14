"""Builds report_short from report_cut: the body only (no appendix) and no reference to the appendix.

report_short = report_cut with (i) the appendix removed, (ii) every pointer to the appendix removed or rephrased so that
the result is still stated (the REWRITES list below, exact strings, each one must be found), (iii) the three statements
of the stability subsection of the frequentist analysis that the numbers of the extended report contradict corrected.
Nothing is edited by hand: run make_cut.py first, then this script, then  cd report_short && tectonic main.tex.

usage: python3 report_extended/split_body_appendix/make_short.py
"""
import os, re, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from float_place import place_floats, never_cited, FLOAT as _FLOAT

BUILD = os.path.dirname(os.path.abspath(__file__))                       # report_extended/split_body_appendix
CUT = os.path.abspath(os.path.join(BUILD, '..', '..', 'report_cut'))
OUT = os.path.abspath(os.path.join(BUILD, '..', '..', 'report_short'))
for d in ('sections', 'figures'):
    shutil.rmtree(f'{OUT}/{d}', ignore_errors=True)       # no file of an earlier build survives
    os.makedirs(f'{OUT}/{d}', exist_ok=True)

main = open(CUT + '/main.tex').read()
main = main.replace('\\input{sections/appendix}\n', '')
open(OUT + '/main.tex', 'w').write(main)
for f in ('preamble.tex', 'biblio.bib', 'mainNotes.bib'):
    if os.path.exists(f'{CUT}/{f}'):
        shutil.copy(f'{CUT}/{f}', f'{OUT}/{f}')
inputs = re.findall(r'\\input\{sections/([^}]+)\}', main)

# (file, old, new): every old string must be present exactly once
R = [
 ('abstract', ', described in Appendix~\\ref{sec:ml}, leaves', ' leaves'),
 ('introduction', ' (Appendix~\\ref{sec:limitations})', ''),
 
 ('introduction', '\\ref{sec:bayes_checks} \\\\', '\\apponly{sec:bayes_checks} \\\\'),
 ('introduction', 'App.~\\ref{sec:ml_stability} \\\\\nFrequentist fit, ML', '\\ref{sec:ml_stability} \\\\\nFrequentist fit, ML'),
 ('introduction', 'App.~\\ref{sec:ml_stability} \\\\\nBayesian fit, Dalitz', '\\ref{sec:ml_stability} \\\\\nBayesian fit, Dalitz'),
 ('data', '; the cut flow is in Table~\\ref{tab:cutflow} of Appendix~\\ref{app:ext_data}.', '.'),
 ('model', ', hence the Poisson term, Appendix~\\ref{app:model})', ', hence the Poisson term)'),
 ('frequentist', '(Sec.~\\ref{sec:bayes_toymc}, $65.0', '($65.0'),
 ('frequentist', 'such as the shape of the signal tails (Sec.~\\ref{sec:bayes_gof}).', 'such as the shape of the signal tails.'),
 ('bayesian', '\\texttt{emcee}~\\cite{emcee} (Appendix~\\ref{app:theory}):', '\\texttt{emcee}~\\cite{emcee}:'),
 ('bayesian', '; trace plot in Fig.~\\ref{fig:bayes_trace}, Appendix~\\ref{sec:appendix})', ')'),
 ('bayesian', '(Sec.~\\ref{sec:frequentist}, Appendix~\\ref{sec:appendix_freq})', '(Sec.~\\ref{sec:frequentist})'),
 ('bayesian', 'whose effect on $\\Araw$ is negligible (Appendix~\\ref{sec:bayes_gof}).', 'whose effect on $\\Araw$ is negligible.'),
 ('bayesian', '~\\cite{dickey1971} (Appendix~\\ref{sec:appendix_freq}),', '~\\cite{dickey1971},'),
 ('bayesian', 'as the prior-sensitivity check of Appendix~\\ref{sec:bayes_checks} confirms for $A$', 'as a prior-sensitivity check confirms for $A$'),
 ('bayesian', '(Fig.~\\ref{fig:checks_bf}; the Savage-Dickey densities are in Fig.~\\ref{fig:bayes_sd} of Appendix~\\ref{app:ext_bayes}, further checks in Appendix~\\ref{sec:checks_profile}).',
  '(Fig.~\\ref{fig:checks_bf}).'),
 ('systematics', ', Appendix~\\ref{sec:bayes_gof})', ')'),
 ('systematics', ', and all the Bayesian fits in Table~\\ref{tab:bayes_all} of Appendix~\\ref{sec:syst_tables}.', '.'),
 ('systematics', 'The scan of each cut in steps of $0.05$ (Fig.~\\ref{fig:syst_pid_scan} of Appendix~\\ref{app:syst}) shows', 'The scan of each cut in steps of $0.05$ shows'),
 ('systematics', '\n\nThe numbers of the three scans are given in Appendix~\\ref{sec:freq_stab_numbers}.', ''),
 # statements of the stability subsection contradicted by the numbers (extended report: Appendix, "Numbers")
 ('systematics', 'Changing the extremes of the signal region not significant fluctation in $\\Araw$ are observed, suggesting that the sideband preliminar fit could be avoided.',
  'Changing the extremes of the signal region, no significant fluctuation in $\\Araw$ is observed ($6\\times10^{-5}$ at most), suggesting that the sideband preliminary fit could be avoided.'),
 ('systematics', 'This influences significantly the $\\Araw$ estimation, but always within the $68\\%$ CI, confirming the cut selection as reasonable.',
  'On this grid the largest shift is $0.0126$, above the statistical uncertainty of the nominal fit ($0.0085$) but compatible with the fluctuation $\\sigma_B$ expected for nested samples, since the grid contains much smaller samples (mean uncertainty $0.0112$); the scan of one cut at a time above shows no shift beyond $1.3\\,\\sigma_B$ on the tight side.'),
 ('systematics', 'Across all scan configurations, the maximum observed variation in $\\Araw$ remains below the statistical uncertainty ($\\Delta \\Araw < \\sigma_{\\mathrm{stat}}$), demonstrating the stability of the nominal fit result.',
  'The largest shifts of $\\Araw$ are $0.0048$ over the grid of window edges, where the extreme corners are the smallest and most asymmetric windows, and $6\\times10^{-5}$ over the signal region; the preselection remains the dominant contribution, as quantified by the systematic study.'),
 ('validation', ', as done for the Bayesian interval in Appendix~\\ref{sec:bayes_toymc}:', ':'),
 ('validation', 'down to the $3\\sigma$ level (Appendix~\\ref{sec:validation_power}).', 'down to the $3\\sigma$ level.'),
 ('dalitz', 'with the charm vetoes and for other grids (Appendix~\\ref{app:ext_dalitz}).', 'with the charm vetoes and for other grids.'),
 ('dalitz', '$-0.145\\pm0.019$ (Fig.~\\ref{fig:dalitz_phi} of Appendix~\\ref{app:ext_dalitz}).', '$-0.145\\pm0.019$.'),
 ('dalitz', 'sWeight background subtraction (Appendix~\\ref{sec:limitations}),', 'sWeight background subtraction,'),
 ('comparison', 'a machine-learning selection (Appendix~\\ref{sec:ml}) confirms', 'a machine-learning selection confirms'),
 ('comparison', 'under the signal (Appendix~\\ref{sec:ml}).', 'under the signal.'),
 ('comparison', 'one-sided, Appendix~\\ref{sec:appendix_freq})', 'one-sided)'),
 ('comparison', 'toy MC (Appendix~\\ref{sec:validation_power}).', 'toy MC (Sec.~\\ref{sec:validation}).'),
 ('comparison', '(Appendix~\\ref{sec:ml_stability}, Fig.~\\ref{fig:ml_three}).', '.'),
 ('comparison', 'to remove; the classifier is described in Appendix~\\ref{sec:ml}.', 'to remove.'),
 ('conclusions', 'a machine-learning selection (Appendix~\\ref{sec:ml}) confirms', 'a machine-learning selection confirms'),
 ('conclusions', 'bound the result (Appendix~\\ref{sec:limitations}):', 'bound the result:'),
]
# ---- structural edits of the short version (figures replaced, removed or added; Sec. "Comparison" merged into the conclusions)
OVR = os.path.join(BUILD, 'short_overrides')
app = open(CUT + '/sections/appendix.tex').read()
FIGRE = r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}'


def float_block(source, label):
    for m in re.finditer(FIGRE, source, re.S):
        if '\\label{' + label + '}' in m.group(0):
            return m.group(0)
    sys.exit('figure not found: ' + label)


def drop_float(t, label):
    b = float_block(t, label)
    return t.replace(b + '\n', '').replace(b, '')


inputs = [n for n in inputs if n != 'comparison']                  # merged into the conclusions
main = main.replace('\\input{sections/comparison}\n', '')
open(OUT + '/main.tex', 'w').write(main)
text = {n: open(f'{CUT}/sections/{n}.tex').read() for n in inputs}
for n in os.listdir(OVR):
    if n.endswith('.tex'):
        text[n[:-4]] = open(f'{OVR}/{n}').read()
_FL = r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}'
for _m in set(re.findall(r'%%FIG:([^%]+)%%', text.get('ml', ''))):
    _blk = [m_.group(0) for m_ in re.finditer(_FL, app, re.S) if '\\label{' + _m + '}' in m_.group(0)][0]
    _blk = _blk.replace('Appendix~\\ref', 'Sec.~\\ref').replace('App.~\\ref', 'Sec.~\\ref')
    text['ml'] = text['ml'].replace(f'%%FIG:{_m}%%', _blk)

bad = 0
R = [r for r in R if r[0] not in ('comparison', 'conclusions', 'abstract', 'frequentist', 'validation')]     # replaced by short_overrides/*.tex
import difflib
_app_labels = set(re.findall(r'\\label\{([^}]+)\}', app))
_body_labels = set(l for _t in text.values() for l in re.findall(r'\\label\{([^}]+)\}', _t))
def _macro(labs):
    return '\\appref{%s}' % labs[0] if len(labs) == 1 else ('\\apprefs{%s}{%s}' % tuple(labs[:2]) if len(labs) == 2 else ''.join('\\appref{%s}' % l for l in labs))
def _with_pointer(old_s, new_s):
    """new_s plus \\appref where old_s pointed to the appendix"""
    labs = []
    for m in re.finditer(r'\\ref\{([^}]+)\}', old_s):
        if m.group(1) in _app_labels and m.group(1) not in _body_labels and m.group(1) not in labs and not m.group(1).startswith(('fig:', 'tab:')):
            labs.append(m.group(1))
    if not labs or new_s.count('\\appref') or new_s.count('\\secorapp') or new_s.count('\\apponly'):
        return new_s
    sm = difflib.SequenceMatcher(None, old_s, new_s, autojunk=False)
    pos = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('delete', 'replace') and re.search(r'ref\{|Appendix|App\.', old_s[i1:i2]):
            pos = j2 if tag == 'replace' else j1
    if pos is None:
        return new_s
    return new_s[:pos] + _macro(labs) + new_s[pos:]
R = [(n, o, _with_pointer(o, w)) for n, o, w in R]
for n, old, new in R:
    if text[n].count(old) != 1:
        print(f'NOT UNIQUE/FOUND in {n}: {old[:70]!r} ({text[n].count(old)})'); bad += 1; continue
    text[n] = text[n].replace(old, new)
# ---- structural edits (after the rewrites above, which act on the original strings)
sd = float_block(app, 'fig:bayes_sd')
MOVED = {'sec:bayes_gof': float_block(text['bayesian'], 'fig:bayes_fit'),                          # figures of the old body that go to the appendix
         'app:dalitz_phi': float_block(text['dalitz'], 'fig:dalitz_region'),
         'sec:freq_stab_numbers': float_block(text['systematics'], 'fig:freq_stability')}
text['bayesian'] = text['bayesian'].replace(float_block(text['bayesian'], 'fig:checks_bf'), sd)   # BF vs prior width -> Savage-Dickey
text['bayesian'] = re.sub(r'and the conclusion is unchanged \(Fig\.~\\ref\{fig:checks_bf\}\)((?:\\apprefs?\{[^}]*\}(?:\{[^}]*\})?)*)\.', r'and the conclusion is unchanged\1.', text['bayesian'])
for old, new in (('$BF_{10}\\approx9\\times10^{3}$), decisive', '$BF_{10}\\approx9\\times10^{3}$; Fig.~\\ref{fig:bayes_sd}a), decisive'),
                 ('yet $BF_{01}(A_b=0)\\approx3.5$, because', 'yet $BF_{01}(A_b=0)\\approx3.5$ (Fig.~\\ref{fig:bayes_sd}b)\\appref{app:sd_densities}, because'),
                 ('(Fig.~\\ref{fig:checks_bf}, crossing at $W\\approx0.3$)', '(crossing at $W\\approx0.3$)')):
    assert text['bayesian'].count(old) == 1, old
    text['bayesian'] = text['bayesian'].replace(old, new)
_fstab = float_block(text['systematics'], 'fig:freq_stability')
text['dalitz'] = re.sub(r'Three caveats limit the interpretation:.*?carries it\.',
                        lambda m: 'The asymmetries are counts without background subtraction, and part of the local variation can be instrumental' +
                                  ('\\appref{sec:limitations}' if 'appref{sec:limitations}' in m.group(0) else '') + '.',
                        text['dalitz'], count=1, flags=re.S)
text['dalitz'] = drop_float(text['dalitz'], 'fig:dalitz_region')                                  # ordered Dalitz plot: not in the short
for old, new in (('The ordered Dalitz plot (Fig.~\\ref{fig:dalitz_region}a) shows the', 'The Dalitz plane shows the'),
                 (' (spectra of the two charges in Fig.~\\ref{fig:dalitz_region}b, where the $B^-$ peak is visibly smaller)', '')):
    assert text['dalitz'].count(old) == 1, old
    text['dalitz'] = text['dalitz'].replace(old, new)
belt = float_block(app, 'fig:checks_belt')                                                         # confidence belt: added
anchor = '\\subsection{Bias, Efficiency and Coverage}'
assert 'fig:checks_belt' in text['validation']
_belt = float_block(text['validation'], 'fig:checks_belt')                                      # belt -> injection, bias and power
text['validation'] = text['validation'].replace(_belt, float_block(app, 'fig:validation_injection'))
_o = 'Inverting the Neyman construction of the same experiments (Fig.~\\ref{fig:checks_belt}) at the observed $\\hat A$\\appref{sec:checks_belt} gives the $68\\%$ interval $[-0.0528,-0.0351]$, against the Minos interval $[-0.0526,-0.0356]$.'
assert text['validation'].count(_o) == 1
text['validation'] = text['validation'].replace(_o, 'Inverting the Neyman construction of the same experiments at the observed $\\hat A$\\appref{sec:checks_belt} gives the $68\\%$ interval $[-0.0528,-0.0351]$, against the Minos interval $[-0.0526,-0.0356]$; Figure~\\ref{fig:validation_injection} shows bias, pull width and power as a function of the injected asymmetry.')
_ms = float_block(text['data'], 'fig:mass_spectrum')                                          # spectrum: one column
text['data'] = text['data'].replace(_ms, r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/mass_spectrum_col.pdf}
\caption{(a) Reconstructed $B\to KKK$ invariant mass in the nominal fit window $[5100,5600]\,\MeV$ for $B^+$ and $B^-$, logarithmic scale; the signal region $[5200,5350]\,\MeV$ is shaded. (b) Counting asymmetry in $25\,\MeV$ bins of the mass for the two magnet polarities, and for all candidates (dashed); the two polarities agree within their uncertainties and the asymmetry is similar in the peak and in the sidebands.}
\label{fig:mass_spectrum}
\end{figure}""")
text['bayesian'] = re.sub(r'\n?(?:\\paragraph\*\{Further material\} is in Appendix~\\ref\{[^}]+\}\.)? ?\\emph\{Model Adequacy: Goodness of Fit and Posterior Predictive Check\}:[^\n]*(?:\n|$)', '\n', text['bayesian'])
_cb = float_block(text['bayesian'], 'fig:bayes_corner')                                          # corner plot redrawn for the short
text['bayesian'] = text['bayesian'].replace(_cb, r"""\begin{figure*}[t]
\centering
\includegraphics[width=0.78\textwidth]{figures/short_corner.pdf}
\caption{Marginal posteriors and correlations of the six physics parameters of the nominal fit (the six background-shape parameters are marginalized). Diagonal: posterior with its median (line) and $68\%$ interval (band), quoted above as median $\pm$ half-width; below the diagonal: two-dimensional posteriors with the correlation coefficient.}
\label{fig:bayes_corner}
\end{figure*}""")
text['bayesian'] = drop_float(text['bayesian'], 'fig:bayes_fit')                               # same fit as the frequentist figure
_pat = (r'Figure~\\ref\{fig:bayes_fit\} compares the nominal model with the data of the two charges: it follows the spectrum over two orders of '
        r'magnitude, and the pulls show signal tails heavier than a Gaussian, whose effect on \$\\Araw\$ is negligible((?:\\apprefs?\{[^}]*\}(?:\{[^}]*\})?)*)\.')
assert len(re.findall(_pat, text['bayesian'])) == 1
text['bayesian'] = re.sub(_pat, lambda m: 'The nominal model with the posterior medians coincides with the fit of Fig.~\\ref{fig:freq_jointfit}, which it follows over two orders of magnitude' + m.group(1) + '.', text['bayesian'])
# ---- figures added to the short: shifts of all variants, goodness of fit with toys, phi band, three methods with ML, profile vs posterior
_cmp = open(CUT + '/sections/comparison.tex').read()
_sv = float_block(app, 'fig:syst_pid_scan'); _gof = float_block(app, 'fig:validation_gof'); _phi = float_block(app, 'fig:dalitz_phi')
_ml3 = float_block(app, 'fig:ml_three'); _prof = float_block(_cmp, 'fig:checks_profile')
_prof = r"""\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/short_profile_polarity.pdf}
\caption{(a) Frequentist profile likelihood, $\exp[-q(A)/2]$ normalized, and Bayesian marginal posterior of $A$: the two coincide. (b) Profile likelihood $q(A)$ of the two magnet polarities (parabolic fits to the points), their sum and the joint fit: the likelihoods of the two polarities combine into that of the full sample.}
\label{fig:checks_profile}
\end{figure*}"""
# ---- lighter table of the 13 variants: one column per method, "shift ($\sigma_B$)", no bold, no sample size
_tab = float_block(text['systematics'], 'tab:syst')
_rows = []
for _l in _tab.split('\n'):
    if '&' in _l and _l.rstrip().endswith('\\\\') and 'multicolumn' not in _l and not _l.lstrip().startswith('Group &'):
        _c = [re.sub(r'\\mathbf\{([^}]*)\}', r'\1', x.strip().rstrip('\\').strip()) for x in _l.rstrip().rstrip('\\').split('&')]
        if len(_c) == 7:
            def _cell(d, sb):
                d = d.strip('$'); sb = sb.strip('$')
                return f'${d}$' if sb == '--' else f'${d}$ (${sb}$)'
            _rows.append((_c[0], _c[1], _cell(_c[3], _c[4]), _cell(_c[5], _c[6])))
_GU = {'Window': ('0.0014', '0.0014'), 'Background': ('0.0037', '0.0040'), 'Signal': ('0.0002', '0.0006'), 'Preselection': ('0.0058', '0.0058')}
_body = ''
_last = None
for g, v, b, f in _rows:
    if g and _last is not None:
        _body += f'& \\emph{{group uncertainty}} & $\\mathit{{{_GU[_last][0]}}}$ & $\\mathit{{{_GU[_last][1]}}}$ \\\\\n\\midrule\n'
    _body += f'{g} & {v} & {b} & {f} \\\\\n'
    _last = g or _last
_body += f'& \\emph{{group uncertainty}} & $\\mathit{{{_GU[_last][0]}}}$ & $\\mathit{{{_GU[_last][1]}}}$ \\\\\n'
_new_tab = r"""\begin{table}[t]
\centering
\small
\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{\begin{tabular}{llcc}
\toprule
Group & Variant & Bayesian & Frequentist \\
\midrule
""" + _body + r"""\midrule
\multicolumn{2}{l}{Total (quadrature)} & $0.0070$ & $0.0072$ \\
\bottomrule
\end{tabular}}
\caption{Systematic study common to the Bayesian and frequentist fits: shift $\Delta$ of $\Araw$ for each variant and, in parentheses and for the variants that change the sample, the expected statistical spread $\sigma_B$ of the shift. The largest contributing shift of each group is its uncertainty (italics); the groups are added in quadrature. The single exponential is rejected by the data and excluded. For the window $[5150,5600]$ the frequentist uncertainty is scaled from the number of events, since the Hesse matrix of that fit is not accurate.}
\label{tab:syst}
\end{table}"""
text['systematics'] = text['systematics'].replace(_tab, _new_tab)
MOVED['sec:syst_tables'] = _new_tab
_sv = float_block(app, 'fig:syst_pid_scan')
text['systematics'] = text['systematics'].replace('\\label{tab:syst}\n\\end{table}', '\\label{tab:syst}\n\\end{table}\n\nFigure~\\ref{fig:syst_pid_scan} shows the scan of each particle-identification cut, with the expected spread $\\sigma_B$.\n\n' + _sv, 1)
_i = text['systematics'].index('\\subsection{Stability Checks of the Frequentist Fit}')          # numbers go to the additional results
text['systematics'] = text['systematics'][:_i].rstrip() + '\n'

# ---- every result of the appendix of the cut, as text: "Additional results" at the end of the matching section
def _braced(t, i):
    """text of the {...} group whose opening brace is at t[i]"""
    d = 0
    for j in range(i, len(t)):
        if t[j] == '{':
            d += 1
        elif t[j] == '}':
            d -= 1
            if d == 0:
                return t[i + 1:j]
    return t[i + 1:]


SKIP = ('Hesse Uncertainties', 'Sampling the Posterior', 'Posterior Summaries and Model Comparison', 'Flat Priors and the Evidence',
        'Savage--Dickey Ratio', 'Extended Likelihood', 'Label Switching', 'Trace Plots', 'Posterior of the Raw Asymmetry',
        'Savage--Dickey Densities', 'Bayes Factor of $A=0$', 'All Fits of the Nominal', 'List of the Variants', 'Combination Rule',
        'Results and Particle-Identification Scan', 'Region of the Published Analysis', 'Headline Result', 'Machine-Learning Cross-Check',
        'Machine Learning and the Fits', 'Scope of the Study', 'Lessons of Each Method', 'Systematic Study and Dependence',
        'Background:', 'Description of the Record')
TARGET = {'Dataset and Preselection': 'data', 'Mass Model and Likelihood': 'model', 'Frequentist Analysis': 'frequentist',
          'Bayesian Analysis': 'bayesian', 'Systematic Uncertainties': 'systematics',
          'Validation of the Estimator and of the Intervals': 'validation', 'Local Asymmetry in the Dalitz Plane': 'dalitz',
          'Comparison of the Two Approaches': 'conclusions', 'Conclusions, Scope and Limitations': 'conclusions',
          'Machine-Learning Analysis': 'conclusions'}
extra = {}
for sec in re.split(r'(?m)^(?=\\section\{)', app):
    m = re.match(r'\\section\{([^}]*)\}', sec)
    if not m or m.group(1) not in TARGET:
        continue
    items = []
    for sub in re.split(r'(?m)^(?=\\subsection\{)', sec)[1:]:
        title = re.match(r'\\subsection\{([^}]*)\}', sub).group(1)
        if any(k in title for k in SKIP):
            continue
        k = sub.find('\\emph{In short:')
        if k < 0:
            continue
        sent = _braced(sub, k + len('\\emph')).strip()
        sent = re.sub(r'^In short:\s*', '', sent)
        sent = sent[0].upper() + sent[1:]
        if not sent.endswith('.'):
            sent += '.'
        if f'\\subsection{{{title}}}' in text.get(TARGET[m.group(1)], ''):          # already a subsection of the body
            continue
        if m.group(1) == 'Machine-Learning Analysis':
            title = 'Machine learning --- ' + title
        items.append(f'\\paragraph*{{{title}.}} {sent}\n')
    extra.setdefault(TARGET[m.group(1)], []).extend(items)
# ---- hand-condensed "Additional Results": the same checks and numbers as the appendix, in one paragraph per section
ADD = {
 'data': [
  ("Selection", r"""The kaon-identification requirement on the three tracks is by far the most selective step of the preselection, and only $42$ candidates of the fit window ($0.2\%$) share a track with another candidate; all are kept.""")],
 'model': [
  ("Minimal model", r"""A six-parameter model on a narrower window gives $\Araw=-0.0497$ ($68\%$ CI $[-0.058,-0.041]$), compatible with the nominal result within its uncertainty; it is the model of the low-statistics comparison.""")],
 'frequentist': [
  ("Neyman construction and momentum", r"""Inverting the test of $A_{\mathrm{true}}$ with toy-MC experiments gives the same interval as the likelihood ratio, and $\Araw$ does not depend on the momentum of the $B$."""),
  ("Uncertainties of the twelve parameters", r"""The standard deviations from the Hesse matrix, the bootstrap of the data and the posterior agree for $A$, $A_b$, $\mu$ and $\sigma$; the bootstrap differs for the two yields, because it keeps the total number of candidates fixed, and for the weakly constrained steep slope $\lambda_1^+$."""),
  ("Starting values", r"""$29$ of $30$ fits from random starting points reach the same minimum.""")],
 'bayesian': [
  ("Sampler", r"""An own Metropolis-Hastings sampler reproduces the posterior of the ensemble sampler."""),
  ("Intervals", r"""Equal-tailed and highest-posterior-density $68\%$ intervals coincide for $A$ and $A_b$ and differ for the skewed parameters; the profile likelihood is tabulated for combination with other results."""),
  ("Priors", r"""No change of the prior on the nuisance parameters or of its parametrization moves $A$ by more than $0.0002$, and the evidence by direct integration agrees in order of magnitude with the Savage-Dickey ratio."""),
  ("Model adequacy", r"""The model does not reach $\chi^2/\mathrm{ndof}\approx1$ ($164.1/88$), because the signal tails are heavier than a Gaussian, but $\Araw$ changes by $-0.0002$ with a Crystal Ball signal; the posterior predictive check reproduces the asymmetry in slices of the mass ($p_B=0.67$) and puts the difference of the $\chi^2$ of the two charges at the edge of the replicas ($p_B=0.08$)."""),
  ("Robustness and coverage", r"""$\Araw$ agrees between the magnet polarities ($0.7\sigma$) and the $D^0$ veto moves it by $2.3$ times the expected spread; $200$ toy-MC experiments give coverage $0.65\pm0.03$ ($68\%$) and $0.93\pm0.02$ ($95\%$) with pulls of width $1.04$.""")],
 'systematics': [
  ("Checks", r"""The $\sigma_B$ rule is verified by bootstrap (the spread of the shift between a sample and its tighter subsample is $\sigma_B$), and the grid scans of the Frequentist fit move $\Araw$ by at most $0.0126$ (PID cuts, smaller samples), $0.0048$ (window edges) and $6\times10^{-5}$ (signal region).""")],
 'validation': [
  ("Asymptotic behavior", r"""With the number of candidates the bias tends to zero, the spread scales as $N^{-1/2}$ and the distribution of $\hat A$ becomes Gaussian."""),
  ("Goodness of fit and binning", r"""The binned $\chi^2$ of toy experiments follows its asymptotic distribution, so its $p$-values are accurate: the misfit of the data is $5.4$ standard deviations in $10\,\MeV$ bins and $2.0$ in $2.5\,\MeV$ bins, because it lives on the scale of the flanks of the peak.""")],
 'dalitz': [
  ("Null tests", r"""Permuting the charges gives $\chi^2=31.5\pm8.4$ and never reaches $80.8$ in $2000$ permutations, and the phase-space simulation is consistent with no asymmetry (slightly optimistic $p$-values).""")],
 'conclusions': [
  ("Numerical agreement", r"""The estimates of the two fits differ by $0.0002$, both uncertainties are $0.0085$, and sideband subtraction gives $-0.0503\pm0.0083$."""),
  ("Profile, posterior and polarities", r"""Profiling and marginalizing the eleven nuisance parameters give the same distribution of $A$, since $A$ is almost uncorrelated with them; the two polarities agree and their combined likelihood reproduces the joint fit; the $68\%$ and $95\%$ regions in $(A,A_b)$ of the profile ($\Delta\mathrm{NLL}$ at $2.30$ and $5.99$) and of the highest posterior density coincide, with $A=0$ outside both."""),
  ("Tests and goodness of fit", r"""The exact binomial test on all candidates gives $6.1\sigma$. The same fit has $p=0.03$ in $2.5\,\MeV$ bins and $p\approx5\times10^{-6}$ in $10\,\MeV$ bins: the $p$-value of the binned $\chi^2$ depends on the binning as much as on the model."""),
  ("Low statistics", r"""Credible and confidence intervals coincide in value and coverage from $\sim2\times10^4$ down to $50$ candidates."""),
  ("Published analysis", r"""It has $22{,}119\pm164$ signal candidates, a Cruijff signal and misidentified and partially reconstructed backgrounds; its corrections move $-0.060$ to $-0.043\pm0.009\pm0.003\pm0.007$ and cannot be assumed for another selection."""),
  ("Machine learning: hyperparameters and features", r"""The hyperparameter search gains at most $0.005$ in AUC, less than its uncertainty, and the cross-validated maximum overestimates the test AUC (winner's curse); the Random Forest importance and the Lasso coefficients of the features agree only partly (Spearman correlation $0.42$) and both rank the two vertex variables of the $B$ among the top five.""")],
}

ADD_REFS = {
 ('data', 'Selection'): ['app:cutflow', 'app:ext_data'],
 ('model', 'Minimal model'): ['sec:bayes_minimal'],
 ('frequentist', 'Neyman construction and momentum'): ['sec:checks_belt'], ('frequentist', 'Uncertainties of the twelve parameters'): ['sec:checks_covariance'],
 ('frequentist', 'Starting values'): ['sec:checks_multistart'],
 ('bayesian', 'Convergence'): ['sec:appendix', 'sec:checks_mh'], ('bayesian', 'Intervals'): ['sec:checks_hpd'],
 ('bayesian', 'Background asymmetry'): ['app:sd_densities'], ('bayesian', 'Priors'): ['sec:checks_prior'],
 ('bayesian', 'Model adequacy'): ['sec:bayes_gof', 'sec:checks_ppc'], ('bayesian', 'Robustness and coverage'): ['sec:bayes_checks', 'sec:bayes_toymc'],
 ('systematics', 'Variants'): ['sec:syst_tables', 'app:syst_variants'], ('systematics', 'PID scan'): ['app:syst'],
 ('systematics', 'Checks'): ['sec:checks_barlow', 'sec:freq_stab_numbers'],
 ('validation', 'Asymptotic behavior'): ['sec:checks_scaling'], ('validation', 'Goodness of fit and binning'): ['sec:validation_gof'],
 ('dalitz', 'Local deviations'): ['app:ext_dalitz'], ('dalitz', 'Null tests'): ['sec:dalitz_null', 'app:dalitz_perm'],
 ('dalitz', 'The $\\phi(1020)$ band'): ['app:dalitz_phi'],
 ('conclusions', 'Numerical agreement'): ['sec:comparison_numbers'], ('conclusions', 'Profile, posterior and polarities'): ['sec:checks_profile', 'sec:checks_region'],
 ('conclusions', 'Tests and goodness of fit'): ['app:ext_comp'], ('conclusions', 'Low statistics'): ['sec:bayes_lowstat'],
 ('conclusions', 'Published analysis'): ['app:ext_comp'],
 ('conclusions', 'Machine learning: hyperparameters and features'): ['sec:ml_hyper', 'sec:ml_models'],
}
def _ptr(labs):
    return '' if not labs else ('\\appref{%s}' % labs[0] if len(labs) == 1 else '\\apprefs{%s}{%s}' % tuple(labs[:2]))
extra = {k: ['\n\n'.join('\\emph{%s.} %s' % (t, (lambda y, q: (y[:-1] + q + '.') if y.endswith('.') else y + q)(x.rstrip(), _ptr(ADD_REFS.get((k, t), [])))) for t, x in v) + '\n'] for k, v in ADD.items()}
text['systematics'] = open(os.path.join(BUILD, 'short_late', 'systematics.tex')).read()      # compact result and systematics
for n, items in extra.items():
    head = '\\subsection{Additional Results}\n' if n != 'ml' else '\\section{Machine-Learning Analysis}\n\\label{sec:ml_short}\n'
    body = head + '\n'.join(items)
    if n == 'ml':
        text['ml'] = body
    else:
        text[n] = text[n].rstrip() + '\n\n' + body
if 'ml' in text:
    main = open(OUT + '/main.tex').read().replace('\\input{sections/conclusions}', '\\input{sections/ml}\n\\input{sections/conclusions}')
    open(OUT + '/main.tex', 'w').write(main)
    inputs = [n for n in inputs if n != 'ml'] + ['ml']
# references to anything that is not in the short: drop the parenthesis or the word
_labels = set(l for t in text.values() for l in re.findall(r'\\label\{([^}]+)\}', t))
def _clean(t):
    t = re.sub(r'\n?\\paragraph\*\{Further material\} is in (?:Appendix~\\ref\{[^}]+\}|the complete report)\.\n?', '\n', t)
    t = re.sub(r'\s*\((?:[^()]|\([^()]*\))*?(?:Appendix|App\.)[^()]*\)', '', t)
    t = re.sub(r'(?:Appendix|App\.)~\\ref\{[^}]+\}', 'the complete report', t)
    def _fix(m):
        return m.group(0) if m.group(2) in _labels else ''
    t = re.sub(r'\s*\((?:see )?(?:Figs?|Tables?|Tab|Secs?)\.?~\\ref\{([^}]+)\}(?:[^()]|\([^()]*\))*?\)', lambda m: m.group(0) if m.group(1) in _labels else '', t)
    return t
for n in extra:
    text[n] = _clean(text[n])
_q = text['conclusions'].index('\n\n', text['conclusions'].index('The numbers coincide'))
text['conclusions'] = text['conclusions'][:_q] + ' Figure~\\ref{fig:checks_profile} shows the profile likelihood and the marginal posterior of $A$ on the same axis, and the likelihoods of the two magnet polarities.\n\n' + _prof + text['conclusions'][_q:]
# the "In short" summaries only at the level of the sections, not of the subsections
def _drop_subsection_summaries(t):
    out, pos = [], 0
    for m in re.finditer(r'(\\subsection\{[^}]*\}\s*(?:\\label\{[^}]*\}\s*)*)\\emph\{In short:', t):
        k = m.end() - len('\\emph{In short:')
        out.append(t[pos:m.end() - len('\\emph{In short:')])
        pos = k + len('\\emph') + len(_braced(t, k + len('\\emph'))) + 2
        while pos < len(t) and t[pos] in ' \n':
            pos += 1
    out.append(t[pos:])
    return ''.join(out)
def _promote_summaries(t):
    """a section without a summary of its own gets one made of those of its subsections"""
    m = re.match(r'(\\section\{[^}]*\}\s*(?:\\label\{[^}]*\}\s*)*)', t)
    if not m or t[m.end():].startswith('\\emph{In short'):
        return t
    sents = []
    for k in re.finditer(r'\\emph\{In short:', t):
        sents.append(_braced(t, k.start() + len('\\emph')).replace('In short:', '', 1).strip())
    if not sents:
        return t
    return m.group(1) + '\\emph{In short: ' + ' '.join(x if x.endswith('.') else x + '.' for x in sents) + '}\n' + t[m.end():]
for n in text:
    if n not in ('abstract', 'introduction'):
        text[n] = _promote_summaries(text[n])
# mini-guide to each section: what it does and what it finds, one or two sentences
GUIDE = {
 'data': r"the record, the common preselection ($23{,}895$ candidates) and the fit window ($19{,}731$ candidates); the spectrum peaks at $5284\,\MeV$ for both charges and both polarities, and the counting asymmetry is $-0.043\pm0.007$.",
 'model': r"one extended likelihood shared by the two analyses: a Gaussian signal with common mean and width, a double-exponential background for each charge and $A$ as a parameter.",
 'frequentist': r"the likelihood maximized for the two charges together gives $N_s=15{,}328\pm133$ and $\hat A=-0.0441$ with Minos intervals; $A=0$ is rejected at $5.2\sigma$ and the interval has nominal coverage in $1000$ bootstrap replicas.",
 'bayesian': r"the same likelihood with uniform priors, sampled with MCMC: $\Araw=-0.0443$ ($68\%$ interval $[-0.0528,-0.0358]$) with converged chains; the Bayes factor of $A=0$, $BF_{10}\approx9\times10^3$, depends on the prior width.",
 'systematics': r"thirteen variants in four groups, fitted by both methods on the same candidates, give $0.0070$ (Bayesian) and $0.0072$ (frequentist), dominated by the particle-identification requirement.",
 'validation': r"toy-MC experiments show an unbiased estimator at the minimum variance bound, intervals with nominal coverage and a test of $A=0$ with the nominal size.",
 'dalitz': r"the asymmetry is not uniform over the Dalitz plane ($\chi^2=80.8$ for $31$ degrees of freedom) and is concentrated outside the $\phi(1020)$, in the region of the published analysis: $-0.234\pm0.025$ against $-0.239\pm0.020$.",
 'ml': r"classifiers trained on weak labels from the mass spectrum reach a test AUC of $0.86$; a cut on their score leaves $\Araw$ unchanged, and an adaptive partition of the Dalitz plane shows the pattern of the fixed binning.",
 'conclusions': r"the frequentist and Bayesian analyses give the same $\Araw=-0.044\pm0.009\,(\text{stat})\pm0.007\,(\text{syst})$, $5.2$ standard deviations from zero, with validated intervals; they differ in the meaning of the numbers and of the tests, not in their values.",
}
def _set_guide(t, g):
    m = re.match(r'(\\section\{[^}]*\}\s*(?:\\label\{[^}]*\}\s*)*)', t)
    rest = t[m.end():]
    if rest.startswith('\\emph{In short'):
        k = len('\\emph')
        rest = rest[k + len(_braced(rest, k)) + 2:].lstrip('\n ')
    return m.group(1) + '\\emph{In short: ' + g + '}\n' + rest
for n in text:
    text[n] = _drop_subsection_summaries(text[n])
for n, g in GUIDE.items():
    if n in text:
        text[n] = _set_guide(text[n], g)

for n in text:       # "Further material is in Appendix ..." paragraphs
    text[n] = re.sub(r'\n?\\paragraph\*\{Further material\} is in Appendix~\\ref\{[^}]+\}\.\n?', '\n', text[n])
for n, t in text.items():
    for m in re.finditer(r'Appendix|App\.', t):
        s = t[max(0, m.start() - 60):m.end() + 60].replace('\n', ' ')
        if not re.match(r'\s*%', t[t.rfind('\n', 0, m.start()) + 1:m.start()]):
            print(f'LEFT in {n}: ...{s}...'); bad += 1
if bad:
    sys.exit('make_short: fix the list above')
_labels = set(l for t in text.values() for l in re.findall(r'\\label\{([^}]+)\}', t))
for n in text:
    text[n] = _clean(text[n])
SHORT_FIG = {'bayes_nominal_dalitz_maps': 'short_dalitz_maps', 'bayes_nominal_savage_dickey': 'short_savage_dickey', 'syst_pid_scan': 'short_pid_scan',
             'validation_injection_power': 'short_injection_power', 'ml_nominal_three_methods': 'short_ml_three', 'ml_weak_labels': 'short_ml_labels', 'ml_nominal_kmeans': 'short_ml_kmeans', 'validation_profile_polarity': 'short_profile_polarity'}
for n in text:
    for a, b in SHORT_FIG.items():
        text[n] = text[n].replace(f'figures/{a}.pdf', f'figures/{b}.pdf')
# "frequentist" is written "Frequentist", like "Bayesian" (not inside labels, references, citations, file names)
_PROT = re.compile(r'(\\(?:label|ref|eqref|cite|includegraphics|input)(?:\[[^\]]*\])?\{[^}]*\})')
def _cap(t):
    parts = _PROT.split(t)
    return ''.join(p if _PROT.fullmatch(p) else re.sub(r'\bfrequentist\b', 'Frequentist', p) for p in parts)
for n in text:
    text[n] = _cap(text[n])
# the machine-learning section of the body has its own labels, distinct from those of the appendix (the pointers \appref use the latter)
text['ml'] = re.sub(r'\\label\{(sec:ml(?:_[a-z]+)?)\}', r'\\label{\1_body}', text['ml'])
for n in text:
    text[n] = re.sub(r'(?<!app)\\ref\{(sec:ml(?:_[a-z]+)?)\}', r'\\ref{\1_body}', text[n])

# ---- readability: a few key results in bold and the terms being defined in italics (the same body for the short and the cut)
EMPH = {
 'abstract': [
  (r"$\Araw=-0.0441\pm0.0085\,(\text{stat})\pm0.0072\,(\text{syst})$ (Frequentist) and $-0.0443\pm0.0085\pm0.0070$ (Bayesian), $5.2$ standard deviations from zero",
   r"\bnum{\Araw=-0.0441\pm0.0085\,(\text{stat})\pm0.0072\,(\text{syst})} (Frequentist) and \bnum{-0.0443\pm0.0085\pm0.0070} (Bayesian), \textbf{5.2 standard deviations} from zero"),
  (r"the dominant systematic uncertainty is the particle-identification requirement", r"the dominant systematic uncertainty is the \emph{particle-identification requirement}"),
  (r"we find $-0.234\pm0.025$", r"we find \bnum{-0.234\pm0.025}"),
 ],
 'introduction': [(r"so $\Araw$ is not a measurement of $\ACP$;", r"so $\Araw$ is \emph{not} a measurement of $\ACP$;")],
 'data': [
  (r"The nominal fit window, common to the two analyses, is $[5100,5600]\,\MeV$", r"The \textbf{nominal fit window}, common to the two analyses, is $[5100,5600]\,\MeV$"),
  (r"The counting asymmetry of all candidates", r"The \emph{counting asymmetry} of all candidates"),
 ],
 'model': [
  (r"the parameter of interest (POI,", r"the \emph{parameter of interest} (POI,"),
  (r"The likelihood is extended (the yields", r"The likelihood is \emph{extended} (the yields"),
  (r"can only come from the inference.", r"can only come from \emph{the inference}."),
 ],
 'frequentist': [
  (r"is the nominal one.", r"is the \textbf{nominal} one."),
  (r"The maximum-likelihood estimate is $\hat A=-0.0441$", r"The maximum-likelihood estimate is \bnum{\hat A=-0.0441}"),
  (r"The profile likelihood ratio $q(A)", r"The \emph{profile likelihood ratio} $q(A)"),
  (r"$5.20\sigma$", r"\bnum{5.20\sigma}"),
  (r"The fraction is $68.4\pm1.5\%$ against", r"The fraction is \bnum{68.4\pm1.5\%} against"),
 ],
  'bayesian': [
  (r"[width=0.95\textwidth]{figures/short_savage_dickey.pdf}", r"[width=0.78\textwidth]{figures/short_savage_dickey.pdf}"),
  (r"$BF_{01}\approx1.1\times10^{-4}$", r"\bnum{BF_{01}\approx1.1\times10^{-4}}"),
  (r"decisive evidence against $A=0$", r"\textbf{decisive evidence} against $A=0$"),
  (r"Of the Bayesian results only the Bayes factor depends on the prior;", r"Of the Bayesian results \emph{only the Bayes factor} depends on the prior;"),
 ],
 'validation': [
  (r"the estimator is unbiased and reaches the minimum variance bound.", r"the estimator is \textbf{unbiased} and reaches the \textbf{minimum variance bound}."),
  (r"rejects in $0.051\pm0.013$ of the experiments, the nominal size,", r"rejects in $0.051\pm0.013$ of the experiments, the \textbf{nominal size},"),
  (r"an asymmetry below $\sim0.02$ is not resolved", r"an asymmetry below $\sim0.02$ is \emph{not} resolved"),
 ],
 'ml': [
  (r"derived from the mass spectrum, following", r"derived from the mass spectrum (\emph{weak labels}), following"),
  (r"The raw asymmetry is stable (", r"The raw asymmetry is \textbf{stable} ("),
  (r"an unsupervised partition", r"an \emph{unsupervised} partition"),
 ],
 'dalitz': [
  (r"confirms that the local asymmetry is not uniform;", r"confirms that the local asymmetry is \textbf{not uniform};"),
  (r"The negative asymmetry is not produced by the $\phi(1020)$", r"The negative asymmetry is \emph{not} produced by the $\phi(1020)$"),
  (r"gives $\Araw=-0.234\pm0.025$ (Bayesian)", r"gives $\Araw=\bnum{-0.234\pm0.025}$ (Bayesian)"),
 ],
 'systematics': [
  (r"The preselection dominates,", r"The \textbf{preselection} dominates,"),
  (r"shows to be a fluctuation:", r"shows to be a \emph{fluctuation}:"),
 ],
 'conclusions': [
  (r"The numbers coincide, the statements do not.", r"\textbf{The numbers coincide, the statements do not.}"),
  (r"The credible interval states", r"The \emph{credible interval} states"),
  (r"the confidence interval $[-0.0526,-0.0356]$ is", r"the \emph{confidence interval} $[-0.0526,-0.0356]$ is"),
 ],
}
for n_, pairs_ in EMPH.items():
    if n_ not in text:
        continue
    for old_, new_ in pairs_:
        if text[n_].count(old_) != 1:
            print('EMPH: not found once in', n_, ':', old_[:60], text[n_].count(old_))
            continue
        text[n_] = text[n_].replace(old_, new_, 1)
for n in text:
    text[n] = place_floats(text[n])
_nc = never_cited(text)
if _nc:
    print('floats not cited in the text:', _nc)
for n, t in text.items():
    open(f'{OUT}/sections/{n}.tex', 'w').write(t)
alltex = ''.join(text.values())
used = set(re.findall(r'includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', alltex))
for u in used:
    src = f'{CUT}/{u}'
    if not os.path.exists(src):
        src = os.path.abspath(os.path.join(BUILD, '..', u))                  # figure used only by the short
    if os.path.exists(src):
        shutil.copy(src, f'{OUT}/{u}')
    else:
        print('missing figure', u)
print('report_short built:', len(used), 'figures,', len(inputs), 'body files')

# ---- macros of the pointers to the appendix, in the short (nothing) and in the cut (Appendix~\ref)
MAC_SHORT = ('\\usepackage{xspace}\\newcommand{\\appref}[1]{}\\newcommand{\\apprefs}[2]{}\\newcommand{\\secorapp}[2]{\\ref{#1}}'
             '\\newcommand{\\apponly}[1]{--}\\newcommand{\\bnum}[1]{\\textbf{\\boldmath$#1$}}\n')
MAC_CUT = ('\\usepackage{xspace}\\newcommand{\\appref}[1]{\\unskip~(Appendix~\\ref{#1})\\xspace}\\newcommand{\\apprefs}[2]{\\unskip~(Appendices~\\ref{#1} and~\\ref{#2})\\xspace}'
           '\\newcommand{\\secorapp}[2]{App.~\\ref{#2}}\\newcommand{\\apponly}[1]{App.~\\ref{#1}}\\newcommand{\\bnum}[1]{\\textbf{\\boldmath$#1$}}\n')
_ORDER = ['introduction', 'data', 'model', 'frequentist', 'bayesian', 'validation', 'ml', 'dalitz', 'systematics', 'conclusions']
main = open(OUT + '/main.tex').read()
main = re.sub(r'\\input\{sections/introduction\}.*?\\input\{sections/conclusions\}\n', lambda m_: ''.join(f'\\input{{sections/{n_}}}\n' for n_ in _ORDER), main, flags=re.S)
open(OUT + '/main.tex', 'w').write(main.replace('\\input{preamble}\n', '\\input{preamble}\n' + MAC_SHORT, 1))
# the cut: the same body, with the pointers, followed by the appendix built by make_cut.py (summaries, figures and tables)
if os.path.isdir(CUT) and os.path.exists(CUT + '/sections/appendix.tex') and not os.environ.get('SHORT_ONLY'):
    cmain = main.replace('\\input{preamble}\n', '\\input{preamble}\n' + MAC_CUT, 1)
    cmain = cmain.replace('\\bibliography{biblio}', '\\input{sections/appendix}\n\n\\bibliography{biblio}', 1)
    open(CUT + '/main.tex', 'w').write(cmain)
    for f in os.listdir(CUT + '/sections'):
        if f.endswith('.tex') and f != 'appendix.tex' and f[:-4] not in text:
            os.remove(f'{CUT}/sections/{f}')
    for n, t in text.items():
        open(f'{CUT}/sections/{n}.tex', 'w').write(t)
    ap = open(CUT + '/sections/appendix.tex').read()
    # a float that is now in the body is not repeated in the appendix (its label points to the body one)
    _body_lab = set(l for t in text.values() for l in re.findall(r'\\label\{([^}]+)\}', t))
    def _dedup(m):
        labs = re.findall(r'\\label\{([^}]+)\}', m.group(0))
        return '' if labs and any(l in _body_lab for l in labs) else m.group(0)
    ap = re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}\n?', _dedup, ap, flags=re.S)
    # the figures that the body shows redrawn are redrawn in the appendix too
    for a_, b_ in SHORT_FIG.items():
        ap = ap.replace(f'figures/{a_}.pdf', f'figures/{b_}.pdf')
    # every float of the appendix is cited where it is presented
    _alltxt = _FLOAT.sub('', '\n'.join(text.values()) + '\n' + ap)
    _refs = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', _alltxt))
    for _m in re.finditer(r'\\(?:appref|apponly)\{([^}]+)\}|\\apprefs\{([^}]+)\}\{([^}]+)\}', _alltxt):
        _refs.update(g for g in _m.groups() if g)
    def _cite(m):
        labs = re.findall(r'\\label\{([^}]+)\}', m.group(0))
        if not labs or any(l in _refs for l in labs):
            return m.group(0)
        l = labs[0]
        kind = 'Table' if m.group(1).startswith('table') else 'Figure'
        if l == 'tab:notation':
            return f'The symbols used in the text are listed in Table~\\ref{{{l}}}.\n\n' + m.group(0)
        return f'The result is shown in {kind}~\\ref{{{l}}}.\n\n' + m.group(0)
    ap = re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}', _cite, ap, flags=re.S)
    for lab_, blk in MOVED.items():
        m_ = re.search(r'\\subsection\{[^}]*\}\s*\\label\{' + re.escape(lab_) + r'\}', ap)
        if not m_:
            print('appendix subsection not found for', lab_); continue
        k_ = re.search(r'\n\\(?:subsection|section)\{', ap[m_.end():])
        k_ = m_.end() + (k_.start() if k_ else len(ap) - m_.end())
        ap = ap[:k_].rstrip('\n') + '\n\n' + blk + '\n' + ap[k_:]
    # every float of the appendix is cited where it is presented
    _alltxt = _FLOAT.sub('', '\n'.join(text.values()) + '\n' + ap)
    _refs = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', _alltxt))
    for _m in re.finditer(r'\\(?:appref|apponly)\{([^}]+)\}|\\apprefs\{([^}]+)\}\{([^}]+)\}', _alltxt):
        _refs.update(g for g in _m.groups() if g)
    def _cite(m):
        labs = re.findall(r'\\label\{([^}]+)\}', m.group(0))
        if not labs or any(l in _refs for l in labs):
            return m.group(0)
        l = labs[0]
        kind = 'Table' if m.group(1).startswith('table') else 'Figure'
        if l == 'tab:notation':
            return f'The symbols used in the text are listed in Table~\\ref{{{l}}}.\n\n' + m.group(0)
        return f'The result is shown in {kind}~\\ref{{{l}}}.\n\n' + m.group(0)
    ap = re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}', _cite, ap, flags=re.S)
    dense = ('\\appendix\n\\makeatletter\\renewcommand\\subsection{\\@startsection{subsection}{2}{\\z@}'
             '{1.1ex plus .3ex minus .2ex}{-0.6em}{\\normalfont\\bfseries}}\\makeatother\n')
    if '\\renewcommand\\subsection{' not in ap:
        ap = ap.replace('\\appendix\n', dense, 1)
    open(CUT + '/sections/appendix.tex', 'w').write(ap)
    for u in used:                      # the figures of the new body
        src = f'{OUT}/{u}'
        if os.path.exists(src):
            os.makedirs(os.path.dirname(f'{CUT}/{u}'), exist_ok=True)
            shutil.copy(src, f'{CUT}/{u}')
    print('report_cut: body of the short + appendix')
