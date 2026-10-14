"""Builds the complete report (natural order, no body/appendix split) in LHCb/report_extended.

Sources: report_extended/sections (body sections and the detailed appendix sections), short_overrides (text of the
frequentist section and of the conclusions, rewritten) and the figures redrawn for the short version.
Every detailed subsection of the former appendix is placed after the body of its own section; the machine-learning
analysis becomes a section of its own. Nothing in the numbers is changed.

usage: python3 report_build/split_body_appendix/build_full.py
"""
import os, re, shutil, sys

BUILD = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(BUILD, '..'))
OUT = os.path.abspath(os.path.join(SRC, '..', 'report_extended'))      # the complete report; SRC (report_build) holds the sources
for d in ('sections', 'figures'):
    shutil.rmtree(f'{OUT}/{d}', ignore_errors=True)
    os.makedirs(f'{OUT}/{d}')


def read(n, base=None):
    t = open(f'{base or SRC + "/sections"}/{n}.tex').read()
    return re.sub(r'(?m)^%.*\n', '', t)


def braced(t, i):
    d = 0
    for j in range(i, len(t)):
        if t[j] == '{':
            d += 1
        elif t[j] == '}':
            d -= 1
            if d == 0:
                return t[i + 1:j]
    return t[i + 1:]


FIGRE = r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}'

# ------------------------------------------------------------------ appendix of the extended report, by section/subsection
app = read('appendix')
app = app[app.index('\\section{'):]
SECS = {}
for sec in re.split(r'(?m)^(?=\\section\{)', app):
    m = re.match(r'\\section\{([^}]*)\}', sec)
    if not m:
        continue
    parts = re.split(r'(?m)^(?=\\subsection\{)', sec)
    SECS[m.group(1)] = dict(head=parts[0], subs=[(re.match(r'\\subsection\{([^}]*)\}', p).group(1), p) for p in parts[1:]])


def subs_of(title, skip=(), order=()):
    items = [(t, p) for t, p in SECS[title]['subs'] if not any(sk in t for sk in skip)]
    if order:
        key = {t: i for i, (t, _p) in enumerate(items)}
        pos = lambda t: next((i for i, o in enumerate(order) if o in t), len(order) + key[t])
        items.sort(key=lambda it: pos(it[0]))
    return '\n'.join(p.rstrip() + '\n' for _t, p in items)


def app_label(title):
    m = re.search(r'\\label\{([^}]+)\}', SECS[title]['head'])
    return m.group(1) if m else None


def add_alias(body, label):
    """the section keeps the label of the appendix section it absorbs, so that old references resolve"""
    if not label or f'\\label{{{label}}}' in body:
        return body
    return re.sub(r'(\\section\{[^}]*\}\s*\\label\{[^}]+\})', lambda m: m.group(1) + f'\n\\label{{{label}}}', body, count=1)


# ------------------------------------------------------------------ body files (the frequentist section and the conclusions are the rewritten ones)
OVR = os.path.join(BUILD, 'short_overrides')
body = {n: read(n) for n in ('introduction', 'data', 'model', 'bayesian', 'systematics', 'validation', 'dalitz', 'comparison')}
body['frequentist'] = read('frequentist', OVR)
body['conclusions'] = read('conclusions', OVR)

# the frequentist stability checks: the list of the body (contradicted by the numbers) is replaced by the corrected
# subsection, which absorbs the numbers of the appendix
st = body['systematics']
i = st.index('\\subsection{Stability Checks of the Frequentist Fit}')
j = st.index('\\includegraphics', i)
j = st.rfind('\\begin{figure', i, j)
k = st.index('\\end{figure*}', j) + len('\\end{figure*}')
fig_stab = st[j:k]
num = [p for t, p in SECS['Systematic Uncertainties']['subs'] if t.startswith('Stability Scans')][0]
ins = re.search(r'\\emph\{In short:(.*?)\}\n', num, re.S).group(1)
items = num[num.index('\\begin{itemize}'):num.index('\\end{itemize}') + len('\\end{itemize}')]
concl = num[num.index('\\end{itemize}') + len('\\end{itemize}'):].split('\\FloatBarrier')[0].strip()
body['systematics'] = st[:i] + r"""\subsection{Stability Checks of the Frequentist Fit}
\label{sec:freq_syst}
\label{sec:freq_stab_numbers}
\emph{In short:""" + ins + r"""}
Three two-dimensional grid scans were made with the simultaneous fit and the nominal double-exponential background at every point: the lower and upper edges of the fit window ($[5000,5200]$ and $[5350,6100]\,\MeV$, $50\,\MeV$ steps; $55$ of the $80$ fits have a valid minimum and an accurate covariance), the edges of the signal region, and the lower cut on \texttt{ProbK} together with the upper cut on \texttt{ProbPi} (Fig.~\ref{fig:freq_stability}). They are stability checks of the fit, from the same notebook as the nominal fit, and do not enter the systematic uncertainty, whose variants and combination rule are those above.
""" + items + '\n' + concl + '\n\n' + fig_stab + '\n' + st[k:]
# what follows the figure in the body is the list's last sentence and a pointer to the numbers: out
body['systematics'] = re.sub(r'(\\end\{figure\*\}\n)\s*Across all scan configurations.*?\n\nThe numbers of the three scans are given in[^\n]*\n', r'\1', body['systematics'], flags=re.S)

# ------------------------------------------------------------------ which detailed subsections are not repeated (they restate the body)
SKIP = {
 'Dataset and Preselection': ('Description of the Record', 'Branches and Preselection', 'Invariant Mass and Fit Window'),
 'Mass Model and Likelihood': (),
 'Frequentist Analysis': (),
 'Bayesian Analysis': ('Posterior of the Raw Asymmetry', 'Bayes Factor of $A=0$', 'Savage--Dickey Ratio',
                       'Bayes Factor of the Background Asymmetry', 'Flat Priors and the Evidence', 'Trace Plots of the Walkers'),
 'Systematic Uncertainties': ('List of the Variants', 'Combination Rule', 'Results and Particle-Identification Scan',
                              'Stability Scans of the Frequentist Fit: Numbers'),
 'Validation of the Estimator and of the Intervals': ('Bias, Efficiency and Coverage',),
 'Local Asymmetry in the Dalitz Plane': ('Local Deviations and Their Robustness', 'Region of the Published Analysis'),
 'Comparison of the Two Approaches': ('Meaning of the Interval', 'Hypothesis Test', 'Goodness of Fit', 'Nuisance Parameters',
                                      'Published Analysis', 'Machine-Learning Cross-Check'),
 'Machine-Learning Analysis': ('Machine Learning and the Fits',),
 'Conclusions, Scope and Limitations': ('Headline Result', 'Systematic Study and Dependence', 'Scope of the Study'),
}
ORDERS = {
 'Bayesian Analysis': ('Sampling the Posterior', 'Sampling and Convergence Diagnostics', 'Equal-Tailed', 'Posterior Summaries',
                       'Savage--Dickey Densities', 'Priors and Bayes Factors', 'Model Adequacy', 'Posterior Predictive Check with',
                       'Robustness Checks', 'Toy-MC Coverage', 'Metropolis-Hastings Sampler'),
 'Systematic Uncertainties': ('All Fits', 'Shifts of All Variants', 'Scan of the Particle', 'Sensitivity of', 'Validation of the Rule'),
 'Conclusions, Scope and Limitations': ('Lessons of Each Method', 'Limitations and Future Work'),
}
ORDER = [('introduction', None), ('data', 'Dataset and Preselection'), ('model', 'Mass Model and Likelihood'),
         ('frequentist', 'Frequentist Analysis'), ('bayesian', 'Bayesian Analysis'), ('systematics', 'Systematic Uncertainties'),
         ('validation', 'Validation of the Estimator and of the Intervals'), ('dalitz', 'Local Asymmetry in the Dalitz Plane'),
         ('ml', 'Machine-Learning Analysis'), ('comparison', 'Comparison of the Two Approaches'),
         ('conclusions', 'Conclusions, Scope and Limitations')]
text = {}
for name, atitle in ORDER:
    if name == 'ml':
        h = SECS[atitle]['head'].rstrip() + '\n'
        text['ml'] = h + '\n' + subs_of(atitle, SKIP.get(atitle, ()))
        continue
    t = body[name].rstrip() + '\n'
    if atitle:
        t = add_alias(t, app_label(atitle))
        extra = subs_of(atitle, SKIP.get(atitle, ()), ORDERS.get(atitle, ()))
        if extra.strip():
            t += '\n' + extra
    text[name] = t

# ------------------------------------------------------------------ editorial edits of the assembled sections
b = text['bayesian']
b = re.sub(r'\n ?\\emph\{Model Adequacy: Goodness of Fit and Posterior Predictive Check\}:[^\n]*\n', '\n', b)       # restates the subsection
trace = [p for t, p in SECS['Bayesian Analysis']['subs'] if t.startswith('Trace Plots')][0]
trace_fig = re.search(FIGRE, trace, flags=re.S).group(0)
k = b.index('\\subsection{Sampling and Convergence Diagnostics}')
m_ = re.search(r'\n\\subsection\{', b[k + 10:])
k2 = k + 10 + m_.start()
b = b[:k2].rstrip() + '\n\n' + trace_fig + '\n' + b[k2:]
b = b.replace('\\subsection{Sampling the Posterior}', '\\subsection{Metropolis-Hastings and the Ensemble Sampler}')
text['bayesian'] = b
s_ = text['systematics']
s_ = s_.replace('The preselection dominates, through the single variant $\\texttt{ProbK}>0.6$ ($+0.0058$, $1.2\\,\\sigma_B$). A scan in steps',
                'A scan in steps')
s_ = s_.replace(' The background group is second, driven by the charge dependence of the background, and the window third.\n\nThe scan is shown',
                '\n\nThe scan is shown')
s_ = re.sub(r'(\\caption\{Shift of \$\\Araw\$ \(frequentist fit\) as a function of \(a\)[^\n]*?)\}\n(\\label\{fig:syst_pid_scan\})',
            lambda m: r'\caption{Shift of $\Araw$ (Frequentist fit) as a function of (a) the $\mathtt{ProbK}$ lower cut and (b) the $\mathtt{ProbPi}$ upper cut, in steps of $0.05$, with the expected statistical spread of the nested sample ($\pm\sigma_B$ dark, $\pm2\sigma_B$ light); points beyond $\sigma_B$ are in red and the nominal cut is the dashed line. On the tight side of $\mathtt{ProbK}$ the points scatter around zero without trend and the pion requirement is flat; loosening $\mathtt{ProbK}$ moves $\Araw$ monotonically. The variant used in the systematic study, $\mathtt{ProbK}>0.6$, is a $1.2\,\sigma_B$ fluctuation.}' + '\n' + m.group(2),
            s_, flags=re.S)
text['systematics'] = s_
GUIDE = {
 'data': r"the record, the common preselection and the fit window; the spectrum peaks at $5284\,\MeV$ for both charges and both polarities, and the counting asymmetry is $-0.043\pm0.007$.",
 'model': r"one extended likelihood shared by the two analyses: a Gaussian signal with common mean and width, a double-exponential background for each charge and $A$ as a parameter.",
 'frequentist': r"the likelihood maximized for the two charges together gives $N_s=15{,}328\pm133$ and $\hat A=-0.0441$ with Minos intervals; $A=0$ is rejected at $5.2\sigma$ and the interval has nominal coverage in $1000$ bootstrap replicas.",
 'bayesian': r"the same likelihood with uniform priors, sampled with MCMC: $\Araw=-0.0443$ ($68\%$ interval $[-0.0528,-0.0358]$) with converged chains; the Bayes factor of $A=0$, $BF_{10}\approx9\times10^3$, depends on the prior width.",
 'validation': r"toy-MC experiments show an unbiased estimator at the minimum variance bound, intervals with nominal coverage and a test of $A=0$ with the nominal size.",
 'dalitz': r"the asymmetry is not uniform over the Dalitz plane ($\chi^2=80.8$ for $31$ degrees of freedom) and is concentrated outside the $\phi(1020)$, in the region of the published analysis: $-0.234\pm0.025$ against $-0.239\pm0.020$.",
}
for n, g in GUIDE.items():
    m_ = re.match(r'(\\section\{[^}]*\}\s*(?:\\label\{[^}]*\}\s*)*)', text[n])
    if m_ and not text[n][m_.end():].lstrip().startswith('\\emph{In short'):
        text[n] = m_.group(1) + '\\emph{In short: ' + g + '}\n' + text[n][m_.end():]

# ------------------------------------------------------------------ code availability, contributions, notation, abstract
code = read('code')
code = re.sub(r'\\texttt\{Code\\_general/Frequentist/\} \(.*?\), \\texttt\{Code\\_general/Bayesian/\}',
              lambda m: (r'\texttt{Code\_general/Frequentist/} (notebook \texttt{06\_frequentist\_fit\_stability\_systematics}: the data, '
                         r'the simultaneous and separate fits, the profile likelihood and the bootstrap coverage, the stability scans, the systematic '
                         r'study common to the Bayesian analysis, the charm vetoes and the Dalitz plane; \texttt{lhcb\_freq.py} with the functions; '
                         r'\texttt{freq\_validation.py} with the toy-MC experiments, profile likelihoods, confidence belt and bootstrap studies of '
                         r'Sec.~\ref{sec:validation}; \texttt{make\_figures.py} for the figures), \texttt{Code\_general/Bayesian/}'), code, flags=re.S)
text['code'] = code
text['contributions'] = read('contributions')
text['notation'] = re.sub(r'(\\label\{sec:notation\}\n)', r'\1The symbols used in the text are listed in Table~\\ref{tab:notation}.\n\n', read('notation'), count=1)
text['abstract'] = read('abstract')

# ------------------------------------------------------------------ pointers: "Appendix" -> "Sec.", figures redrawn, "frequentist" capital
MAC = ('\\usepackage{xspace}\\newcommand{\\appref}[1]{\\unskip~(Sec.~\\ref{#1})\\xspace}'
       '\\newcommand{\\apprefs}[2]{\\unskip~(Secs.~\\ref{#1} and~\\ref{#2})\\xspace}\n')
FIG = {'frequentist_wilks_coverage': 'short_coverage', 'bayes_nominal_savage_dickey': 'short_savage_dickey',
       'syst_pid_scan': 'short_pid_scan', 'validation_injection_power': 'short_injection_power',
       'validation_profile_polarity': 'short_profile_polarity', 'bayes_nominal_dalitz_maps': 'short_dalitz_maps'}
_PROT = re.compile(r'(\\(?:label|ref|eqref|cite|includegraphics|input|href)(?:\[[^\]]*\])?\{[^}]*\})')


def fix(t):
    t = re.sub(r'\n?\\paragraph\*\{Further material\} is in Appendix~\\ref\{[^}]+\}\.\n?', '\n', t)
    t = re.sub(r' of Appendix~\\ref\{[^}]+\}', '', t)
    t = t.replace('sec:appendix_checks', 'sec:checks_profile')
    t = re.sub(r'(Fig\.~\\ref\{[^}]+\}), (?:Appendix|Sec\.)~\\ref\{sec:appendix\}', r'\1', t)
    t = re.sub(r'Sec\.~\\ref\{([^}]+)\}, (?:Appendix|Sec\.)~\\ref\{([^}]+)\}', r'Secs.~\\ref{\1} and~\\ref{\2}', t)
    t = re.sub(r'Appendices~\\ref\{([^}]+)\} and~?\\ref\{([^}]+)\}', r'Secs.~\\ref{\1} and~\\ref{\2}', t)
    t = re.sub(r'Appendix~\\ref\{', 'Sec.~\\\\ref{', t)
    t = re.sub(r'App\.~\\ref\{', 'Sec.~\\\\ref{', t)
    for a, b in FIG.items():
        t = t.replace(f'figures/{a}.pdf', f'figures/{b}.pdf')
    parts = _PROT.split(t)
    return ''.join(p if _PROT.fullmatch(p) else re.sub(r'\bfrequentist\b', 'Frequentist', p) for p in parts)


for n in text:
    text[n] = fix(text[n])

# conclusions: labels already carried by the comparison section, and the \input lines that ended the old appendix
c = text['conclusions']
c = c.replace('\\label{sec:comparison}\n', '', 1).replace('\\label{sec:comparison_published}\n', '', 1)
c = re.sub(r'\n\\input\{sections/(?:code|notation)\}', '', c)
text['conclusions'] = c
# validation: the introduction points to the generation subsection, which comes first; size and power are in their subsection
v = text['validation']
v = re.sub(r'The asymptotic properties of the maximum-likelihood estimator and of the profile-likelihood ratio were also verified on toy MC experiments.*?are excluded\)\.\n',
           'The asymptotic properties of the maximum-likelihood estimator and of the profile-likelihood ratio were verified on toy-MC experiments generated from the fitted model (Sec.~\\\\ref{sec:validation_toys}), which allow the true value of $A$ to be chosen.\n', v, count=1, flags=re.S)
v = re.sub(r'\\emph\{Size and power\.\}.*?\(Sec\.~\\ref\{sec:validation_power\}\)\.\n', 'Together, the estimator is unbiased and efficient, the interval has its nominal coverage and the test its nominal size (Sec.~\\\\ref{sec:validation_power}); the $5.2\\\\sigma$ of the data is an extrapolation of the $\\\\chi^2_1$ tail, which the $293$ experiments check only down to the $3\\\\sigma$ level.\n', v, count=1, flags=re.S)
v = re.sub(r'\\begin\{figure\}\[[a-z!]*\]\n\\centering\n\\includegraphics\[[^\]]*\]\{figures/validation_power\.pdf\}.*?\\end\{figure\}\n', '', v, count=1, flags=re.S)
m1 = re.search(r'\\subsection\{Generation of the Toy-MC Experiments\}.*?(?=\n\\subsection\{)', v, flags=re.S)
gen = m1.group(0)
v = v.replace(gen, '', 1)
k_ = v.index('\\subsection{Bias, Efficiency and Coverage}')
v = v[:k_] + gen.replace('\\subsection{Generation of the Toy-MC Experiments}', '\\subsection{Generation of the Toy-MC Experiments}\n\\label{sec:validation_toys}', 1).rstrip() + '\n\n' + v[k_:]
text['validation'] = v
# dalitz: the homogeneity subsection extends, and does not repeat, the body
d = text['dalitz']
d = re.sub(r'Two of the 32 bins have \$\|S_i\|>3\$, adjacent at.*?it holds in the signal window alone', 'The global homogeneity test, $\\\\chi^2=\\\\sum_iS_i^2=80.8$ for 31 degrees of freedom, holds in the signal window alone',
           d, count=1, flags=re.S) if '(\\$S_i=-3.7\\$ and \\$-3.0\\$, Fig.' in d or True else d
d = re.sub(r'\n\nThe structure is the same with both magnet polarities: in the region of the two significant bins the counting asymmetry is.*?not the different interaction of \$K\^\+\$ and \$K\^-\$ with the detector material\.\n',
           '\n', d, count=1, flags=re.S)
d = d.replace('\\subsection{Homogeneity Test: Signal Window, Vetoes, Grids and Magnet Split}\n',
              '\\subsection{Homogeneity Test: Signal Window, Vetoes, Grids and Magnet Split}\n\\label{sec:dalitz_homog}\n', 1)
d = d.replace('with the charm vetoes and for other grids (Sec.~\\ref{app:ext_dalitz}).', 'with the charm vetoes and for other grids, and with the same structure for both magnet polarities (Sec.~\\ref{sec:dalitz_homog}).')
text['dalitz'] = d

text['introduction'] = re.sub(r'(& -- & )Sec\.~(\\ref\{sec:ml_stability\})', r'\1\2', text['introduction'])
# the mass spectrum of the short (one column, charge and polarity asymmetry) replaces the two-panel figure
ms = re.search(FIGRE.replace('(figure\\*?|table\\*?)', '(figure\\*?)'), text['data'], flags=re.S)
for m in re.finditer(FIGRE, text['data'], flags=re.S):
    if '\\label{fig:mass_spectrum}' in m.group(0):
        text['data'] = text['data'].replace(m.group(0), r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/mass_spectrum_col.pdf}
\caption{(a) Reconstructed $B\to KKK$ invariant mass in the nominal fit window $[5100,5600]\,\MeV$ for $B^+$ and $B^-$, logarithmic scale; the signal region $[5200,5350]\,\MeV$ is shaded. (b) Counting asymmetry in $25\,\MeV$ bins of the mass for the two magnet polarities, and for all candidates (dashed); the two polarities agree within their uncertainties and the asymmetry is similar in the peak and in the sidebands.}
\label{fig:mass_spectrum}
\end{figure}""")
# the coverage figure of the frequentist section is already the new one (override); the validation body keeps its own text

# ------------------------------------------------------------------ main file and output
main = open(f'{SRC}/main.tex').read()
main = main.replace('\\input{preamble}\n', '\\input{preamble}\n' + MAC, 1)
inputs = ['abstract', None]
body_order = ['introduction', 'data', 'model', 'frequentist', 'bayesian', 'systematics', 'validation', 'dalitz', 'ml',
              'comparison', 'conclusions', 'code', 'contributions']
seq = ''.join(f'\\input{{sections/{n}}}\n' for n in body_order) + '\\appendix\n\\input{sections/notation}\n'
main = re.sub(r'\\input\{sections/introduction\}.*?\\input\{sections/appendix\}\n', lambda m: seq, main, flags=re.S)
open(f'{OUT}/main.tex', 'w').write(main)
for n, t in text.items():
    open(f'{OUT}/sections/{n}.tex', 'w').write(t)
shutil.copy(f'{SRC}/preamble.tex', f'{OUT}/preamble.tex')
shutil.copy(f'{SRC}/biblio.bib', f'{OUT}/biblio.bib')
alltex = ''.join(text.values())
used = set(re.findall(r'includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', alltex))
for u in sorted(used):
    src = f'{SRC}/{u}'
    if os.path.exists(src):
        shutil.copy(src, f'{OUT}/{u}')
    else:
        print('missing figure', u)
for f in os.listdir(f'{SRC}/figures'):                      # the scripts that redraw the figures from the saved results
    if f.endswith('.py'):
        shutil.copy(f'{SRC}/figures/{f}', f'{OUT}/figures/{f}')
left = [(n, m.start()) for n, t in text.items() for m in re.finditer(r'Appendix|appendix', t)]
print('report_full built:', len(used), 'figures;', 'leftover "appendix":', len(left), left[:12])
