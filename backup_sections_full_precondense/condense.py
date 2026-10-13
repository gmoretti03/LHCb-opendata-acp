import re,sys,os
os.chdir('/Users/gmoretti/Work/fededegu_martiborsa/LHCb/Report/split_body_appendix/sections_full')
def rd(f): return open(f).read()
def wr(f,t): open(f,'w').write(t)
def para_span(t,start):
    assert t.count(start)==1,(start[:60],t.count(start))
    i=t.index(start); j=t.find('\n\n',i); j=len(t) if j<0 else j
    return i,j
details=[]
def repl(f,start,new,title=None):
    t=rd(f); i,j=para_span(t,start); orig=t[i:j]
    if title: details.append((title,orig))
    wr(f,t[:i]+new+t[j:])
def cut(f,start,title):
    t=rd(f); i,j=para_span(t,start); details.append((title,t[i:j]))
    wr(f,t[:i]+t[j:].lstrip('\n'))

repl('introduction.tex',"In this work we use the LHCb Open Data record",
r"""In this work we use the LHCb Open Data record for $B^{\pm}\to K^{\pm}K^{+}K^{-}$~\cite{opendata_record,opendata_notebook}, a subset of the 2011 data on which the first evidence of an inclusive $CP$ asymmetry in this decay was published~\cite{lhcb_cpv_2013}, to ask: \emph{to what extent do frequentist likelihood fits and Bayesian inference produce consistent, well-controlled estimates of the raw asymmetry $\Araw$?} The aim is a methodological comparison of the two frameworks on the same data, not a competitive measurement of $\ACP$, which cannot be obtained from this record because it lacks the control channel for $A_P$ and $A_D$ (Appendix~\ref{sec:limitations}); the other three channels of the record need a different selection and a model of peaking backgrounds and are not analysed. Section~\ref{sec:data} describes the data, Sec.~\ref{sec:model} the common model, Secs.~\ref{sec:frequentist} and \ref{sec:bayesian} the two fits, Sec.~\ref{sec:syst} the systematic study, Sec.~\ref{sec:validation} the validation, Sec.~\ref{sec:dalitz} the Dalitz plane and Sec.~\ref{sec:comparison} the comparison; a machine-learning selection is in Appendix~\ref{sec:ml}. The main results are collected in Table~\ref{tab:summary}.""",
"Introduction: scope, channels and roadmap")
t=rd('introduction.tex')
a=r" The production asymmetry of $B^\pm$ mesons at the LHC is expected at the level of one per cent; in the published analysis of the same 2011 data, the two corrections, measured with $B^\pm\to J/\psi K^\pm$ decays, together with an acceptance correction, move the asymmetry of $B^\pm\to K^\pm K^+K^-$ from $\Araw=-0.060\pm0.007$ to $\ACP=-0.043\pm0.009\,(\text{stat})\pm0.003\,(\text{syst})\pm0.007\,(J/\psi K)$~\cite{lhcb_cpv_2013}."
assert t.count(a)==1
wr('introduction.tex',t.replace(a," The corrections applied in the published analysis are discussed in Sec.~\\ref{sec:comparison_published}."))

cut('model.tex',"The extended form differs from a mixture","Extended likelihood versus a mixture with fixed total")

repl('bayesian.tex',"The posterior was sampled with the affine-invariant",
r"""The posterior was sampled with the affine-invariant ensemble sampler of Goodman and Weare~\cite{goodman_weare} as implemented in \texttt{emcee}~\cite{emcee} (Appendix~\ref{app:theory}): 32 walkers started around the maximum of the posterior, $25{,}000$ steps with the first $1/6$ discarded as burn-in ($6.7\times10^5$ samples, acceptance $0.38$). The integrated autocorrelation time is $\tau\in[147,183]$ steps, so the chain is $N/\tau\geq113$ autocorrelation times long for every parameter, above the threshold of $50$~\cite{emcee} ($\gtrsim3600$ effectively independent samples). Two runs started uniformly over the whole prior box converge to the same single mode ($\Araw=-0.0446$ and $-0.0442$; trace plot in Fig.~\ref{fig:bayes_trace}, Appendix~\ref{sec:appendix}).""",
"MCMC sampling and convergence (complete)")
repl('bayesian.tex',"The marginal posterior on the raw asymmetry gives",
r"""The marginal posterior on the raw asymmetry gives (median and equal-tailed credible intervals, CI)
\begin{gather*}
\Araw = -0.0443, \\
68\%~\text{CI}~[-0.0528,-0.0358], \\ 95\%~\text{CI}~[-0.0608,-0.0276].
\end{gather*}
The posterior is very close to Gaussian (standard deviation $0.0085$), so equal-tailed and highest-posterior-density intervals coincide; with uniform priors its mode, $-0.0441$, is the maximum-likelihood estimate and the result can be compared directly with the frequentist fit (Sec.~\ref{sec:frequentist}, Appendix~\ref{sec:appendix_freq}). The other physics parameters are $N_s=15{,}328\pm133$, $N_b=4404\pm82$, $A_b=-0.040\pm0.018$, $\mu=5284.39\pm0.16\,\MeV$ and $\sigma=18.00\pm0.14\,\MeV$ (Fig.~\ref{fig:bayes_corner}). $A$ is almost uncorrelated with them; its largest correlation, $-0.17$, is with $A_b$: conditioning on $A_b=0$ moves $A$ to $-0.0475$, as the fit with a charge-symmetric background does ($-0.0471$, Sec.~\ref{sec:syst}). The comparison with the published analysis is in Sec.~\ref{sec:comparison_published}.""",
"Posterior of the raw asymmetry (complete)")
repl('bayesian.tex',"The Bayes factor compares two hypotheses through their evidences",
r"""The Bayes factor compares two hypotheses through their evidences, $BF_{10}=P(D\mid H_1)/P(D\mid H_0)$, and requires proper priors. For the nested hypothesis $H_0$: $A=0$ against $H_1$: $A$ free, with shared priors on the other parameters, it reduces to the Savage-Dickey ratio~\cite{dickey1971} (Appendix~\ref{sec:appendix_freq}),
\[
BF_{01} = \frac{p(A=0\mid D)}{\pi(A=0)}, \qquad \pi(A=0)=\tfrac12 .
\]
$A=0$ lies $5.2$ posterior standard deviations from the median, where the chain has essentially no samples, so $p(A=0\mid D)$ was taken from the Gaussian approximation of the marginal posterior: $BF_{01}\approx1.1\times10^{-4}$ ($BF_{10}\approx9\times10^{3}$), decisive evidence against $A=0$ on Jeffreys' scale~\cite{jeffreys}. The value scales with the prior width (Ockham factor): with $A\in(-0.3,0.3)$, $BF_{01}$ drops by a factor $3.3$ and the conclusion is unchanged (Fig.~\ref{fig:bayes_sd}; further checks in Appendix~\ref{sec:appendix_checks}).""",
"Bayes factor of A=0 (complete)")
repl('bayesian.tex',"The same test on the background asymmetry shows",
r"""The same test on the background asymmetry shows the Ockham factor at work: $A_b=-0.040\pm0.018$ is $2.3$ standard deviations from zero, yet $BF_{01}(A_b=0)\approx3.5$, because the likelihood gain does not compensate for spreading the prior over $(-1,1)$. A negative $A_b$ is plausible given the misidentified $B^\pm\to K^\pm\pi^+\pi^-$ and $\pi^\pm K^+K^-$ decays; $A_b$ is kept free in the nominal model and fixing it to zero is one of the systematic variants (Sec.~\ref{sec:syst}).""",
"Bayes factor of the background asymmetry (complete)")

repl('systematics.tex',"The shift $\\Delta=\\Araw^{\\mathrm{var}}",
r"""The shift $\Delta=\Araw^{\mathrm{var}}-\Araw^{\mathrm{nom}}$ of a variant is not necessarily a systematic effect: when a variant changes the sample (window, preselection) the two samples are nested and part of $\Delta$ is a statistical fluctuation of expected size $\sigma_B=\sqrt{|\sigma_{\mathrm{var}}^2-\sigma_{\mathrm{nom}}^2|}$~\cite{barlow}, and such a variant contributes only if $|\Delta|>\sigma_B$. A variant that the data reject ($\Delta\chi^2>25$ on the same data) is excluded: this removes the single exponential ($\Delta\chi^2=134$, Sec.~\ref{sec:bayes_gof}). The largest remaining $|\Delta|$ of each group is its uncertainty, and the groups are added in quadrature.""",
"Combination rule of the systematic study (complete)")
repl('systematics.tex',"The results are in Table~\\ref{tab:syst}",
r"""The results are in Table~\ref{tab:syst}, and all the Bayesian fits in Table~\ref{tab:bayes_all} of the Appendix. The two methods give the same shifts to within $\sim$$0.0005$ and the same group uncertainties, with totals of $0.0070$ (Bayesian) and $0.0072$ (frequentist):
\begin{gather*}
\text{Bayesian:}\\ \Araw=-0.0443\pm0.0085\,(\text{stat})\pm0.0070\,(\text{syst}),\\[2pt]
\text{frequentist:}\\ \Araw=-0.0441\pm0.0085\,(\text{stat})\pm0.0072\,(\text{syst}).
\end{gather*}
The preselection dominates, through the single variant $\texttt{ProbK}>0.6$ ($+0.0058$, $1.2\,\sigma_B$). A scan in steps of $0.05$ of each cut ($\texttt{ProbK}>0.30$--$0.85$, $\texttt{ProbPi}<0.15$--$0.70$) shows that on the tight side the shifts scatter around zero without a trend, all within $1.3\,\sigma_B$, and the pion requirement is flat: the $\texttt{ProbK}>0.6$ value is a statistical fluctuation and the preselection uncertainty is conservative. Loosening the kaon requirement moves $\Araw$ monotonically ($-0.0019$ to $-0.0111$ for $\texttt{ProbK}>0.45$--$0.30$, the last at $2.5\,\sigma_B$), as expected if misidentified $B\to K\pi\pi$ and $B\to\pi KK$ decays, not in the mass model, enter the sample; this supports the nominal requirement. The background group is second, driven by the charge dependence of the background, and the window third.""",
"Systematic study: results and the particle-identification scan (complete)")
cut('systematics.tex',"Each method computes $\\sigma_B$ from its own uncertainties","Sensitivity of sigma_B to small differences between the methods")
repl('systematics.tex',"\\paragraph*{Numbers.}",
r"""\paragraph*{Numbers.} Three results from the same notebook, which do not enter the systematic uncertainty. Over the grid of window edges the maximum shift is $0.0048$ (mean statistical uncertainty $0.0085$); the signal region, used only to start the sideband fit, moves $\Araw$ by at most $6\times10^{-5}$; on the grid of the two PID cuts the maximum shift is $0.0126$, larger than the nominal $0.0085$ but within the statistical uncertainty of the much smaller samples ($0.0112$ on average), and no shift exceeds $1.3\,\sigma_B$ in the one-cut-at-a-time scan on the tight side. The preselection therefore remains the dominant contribution.""",
"Stability scans of the frequentist fit: numbers (complete)")

repl('validation.tex',"The frequentist results of Sec.~\\ref{sec:frequentist} rest on",
r"""The asymptotic properties of the maximum-likelihood estimator and of the profile-likelihood ratio were also verified on toy MC experiments generated from the fitted model of Sec.~\ref{sec:model} (Poisson yields, masses drawn from the model), as done for the Bayesian interval in Sec.~\ref{sec:bayes_toymc}: $1000$ experiments at the fitted value $A_{\mathrm{true}}=-0.0441$ and $300$ at each of $-0.1$, $-0.02$, $0$ and $+0.05$, each fitted with the nominal frequentist procedure ($1$--$3\%$ of the fits are not flagged valid by \texttt{MIGRAD} and are excluded).""",
"Validation: generation of the toy MC experiments (complete)")
repl('validation.tex',"Table~\\ref{tab:validation_toys} summarizes the estimator",
r"""Table~\ref{tab:validation_toys} summarizes the estimator over the five injected values: the bias of $\hat A$ is at most $0.0008$ ($1.6$ standard errors), its spread equals the mean Hesse uncertainty within $6\%$ and the pulls have width $0.94$--$1.05$, so the estimator is unbiased and reaches the minimum variance bound. At the fitted value the coverage of $t(A_{\mathrm{true}})\leq1$ is $0.696\pm0.015$ ($68.3\%$ level) and $0.954\pm0.007$ ($95.4\%$ level); at $A_{\mathrm{true}}=+0.05$ the $68\%$ coverage is $0.614\pm0.028$, $2.5$ standard deviations low, not unexpected among five values.""",
"Validation: bias, efficiency and coverage (complete)")

repl('dalitz.tex',"Two of the 32 bins have $|S_i|>3$",
r"""Two of the 32 bins have $|S_i|>3$, adjacent at $m^2_{\mathrm{lo}}<4\,\GeV^2$ and $7.5<m^2_{\mathrm{hi}}<11.25\,\GeV^2$ ($S_i=-3.7$ and $-3.0$, Fig.~\ref{fig:bayes_dalitz}); about $0.09$ would be expected if the asymmetry were uniform ($3\times10^{-3}$ for two or more), a look-elsewhere number that refers to any two of the 32 bins. The global homogeneity test, $\chi^2=\sum_iS_i^2=80.8$ for 31 degrees of freedom ($p=3\times10^{-6}$), confirms that the local asymmetry is not uniform; it holds in the signal window alone ($\chi^2=95.6/28$, signal purity $\approx$96\%), with the $D^0$ veto ($79.2/31$) and with all three charm vetoes ($74.4/31$), and for $6\times6$, $10\times10$ and $12\times12$ grids ($81.0/19$, $97.1/43$, $120.0/60$).""",
"Dalitz plane: local deviations and their robustness (complete)")
repl('dalitz.tex',"The ordered Dalitz plot (Fig.~\\ref{fig:dalitz_region}a)",
r"""The ordered Dalitz plot (Fig.~\ref{fig:dalitz_region}a) shows the $\phi(1020)$ band at the kinematic edge ($m^2_{\mathrm{lo}}\approx1.04\,\GeV^2$) and the charm bands of the $D^0$, $J/\psi$ and $\chi_{c0}$. The negative asymmetry is not produced by the $\phi(1020)$: its band ($1964$ candidates) has $A_{\mathrm{count}}=-0.004\pm0.023$, while the significant bins without it have $-0.145\pm0.019$. It sits just above the $\phi$, where the published analysis located it, defining the region $1.2<m^2_{\mathrm{lo}}<2.0\,\GeV^2$, $m^2_{\mathrm{hi}}<15\,\GeV^2$ with $\Araw=-0.239\pm0.020$~\cite{lhcb_cpv_2013}. The region is fixed by the paper, not chosen on our data, so the comparison is unbiased: the nominal model fitted to its $2146$ candidates gives $\Araw=-0.234\pm0.025$ (Bayesian) and $-0.237\pm0.025$ (frequentist, $9.4\sigma$ from zero), reproducing the published value (spectra of the two charges in Fig.~\ref{fig:dalitz_region}b). Caveats: the binning is coarse and was not optimized, the asymmetries are counts without sWeight background subtraction (Appendix~\ref{sec:limitations}), and local variations can be partly instrumental, since the $K^\pm$ interaction asymmetry depends on momenta that are correlated with the Dalitz position. Appendix~\ref{sec:ml_kmeans} revisits the question with an adaptive partition ($K$-means).""",
"Dalitz plane: the region of the published analysis (complete)")

repl('comparison.tex',"\\emph{Hypothesis test.}",
r"""\emph{Hypothesis test.} The exact binomial test of $N^-=N^+$ on all candidates gives $6.1\sigma$ ($A_{\mathrm{count}}=-0.0432\pm0.0071$); it sees signal and background together and does not use the mass. The frequentist test of the fit gives $p=2\times10^{-7}$ ($Z=\sqrt{q(0)}=5.2\sigma$ in the two-sided convention, $5.06\sigma$ one-sided, Appendix~\ref{sec:appendix_freq}), independent of the prior and not the probability of $A=0$; its size and power were verified with toy MC (Sec.~\ref{sec:validation_power}). The $5.2\sigma$ crosses the conventional $5\sigma$ threshold but refers to a raw asymmetry that includes production and detection effects (Sec.~\ref{sec:comparison_published}). The Bayes factor, $BF_{10}\approx9\times10^3$, depends on the prior width (a factor $3.3$ for $A\in(-0.3,0.3)$). For $A$ the two give the same conclusion; for the background asymmetry, $A_b=-0.040\pm0.018$, a frequentist test would call $2.3\sigma$ a hint while $BF_{01}\approx3.5$ mildly favors $A_b=0$ (Sec.~\ref{sec:bayes_factor}): they answer different questions.""",
"Comparison: hypothesis test (complete)")
repl('comparison.tex',"\\emph{Goodness of fit.}",
r"""\emph{Goodness of fit.} The same fit gives $\chi^2/\mathrm{ndof}=441.7/388$ ($p=0.031$) in $2\times200$ bins of $2.5\,\MeV$ and $\approx160/88$ ($p\approx5\times10^{-6}$) in $2\times50$ bins of $10\,\MeV$ (the Bayesian fit, $164.1/88$, uses the second): the misfit is at the $\sim10\,\MeV$ scale and diluted in narrow bins, so the $p$-value is a property of the binning as much as of the model. The posterior predictive check shows the misfit concentrated in the $B^+$ sample (Sec.~\ref{sec:bayes_gof}).""",
"Comparison: goodness of fit (complete)")
repl('comparison.tex',"The published analysis of the same data~\\cite{lhcb_cpv_2013} finds",
r"""The published analysis~\cite{lhcb_cpv_2013} finds $\Araw=-0.060\pm0.007$ with $22{,}119\pm164$ signal candidates, against $-0.0441\pm0.0085$ and $15{,}328\pm133$ here: the Open Data record is a preselected subset and the published model has a Cruijff signal, partially reconstructed four-body decays and a misidentified peaking background from simulation. With the same $D^0$ veto the Bayesian fit gives $-0.0496\pm0.0088$, within $0.93\sigma$ of the published value, although the samples overlap and the selections differ.""",
"Comparison with the published analysis: samples and models (complete)")
repl('comparison.tex',"The raw asymmetry is not the CP asymmetry.",
r"""The raw asymmetry is not the CP asymmetry: $\ACP\approx\Araw-A_P-A_D$ (Eq.~\eqref{eq:acp}), and in the published analysis the corrections move the result from $-0.060$ to $-0.043\pm0.009\pm0.003\pm0.007$ ($3.7\sigma$)~\cite{lhcb_cpv_2013}, by $+0.017$, about two statistical uncertainties of this work. Such a correction cannot be assumed for a different selection, and the $5.2\sigma$ here is the significance of a nonzero raw asymmetry, not of CP violation. A later measurement on a larger data set gives $\ACP=-0.037\pm0.002\pm0.002\pm0.003$ ($8.5\sigma$)~\cite{lhcb_cpv_prd2023,lhcb_cpv_paper}; our result is consistent with these in sign and size.""",
"Comparison with the published analysis: raw versus CP asymmetry (complete)")
repl('comparison.tex',"A classifier trained on weak labels derived from the mass spectrum",
r"""A classifier trained on weak labels from the mass spectrum (test AUC $0.86$) was used as an independent selection: a cut at $0.5$ keeps $92\%$ of the fitted signal and $36\%$ of the background and leaves $\Araw$ within its expected statistical spread ($-0.0451\pm0.0087$ Bayesian, $-0.0447\pm0.0086$ frequentist), without improving the precision; it cannot remove the misidentified decays that peak under the signal. Details, including the $K$-means partition of the Dalitz plane, are in Appendix~\ref{sec:ml}.""",
None)

repl('conclusions.tex',"We have measured the raw charge asymmetry",
r"""We have measured the raw charge asymmetry of $B^\pm\to K^\pm K^+K^-$ decays in the 2011 LHCb Open Data record with a common selection and mass model, and compared how two inference frameworks treat the same question. The frequentist profile-likelihood fit and the Bayesian posterior give $\Araw=-0.0441\pm0.0085\pm0.0072$ and $-0.0443\pm0.0085\pm0.0070$ (statistical, systematic), a nonzero asymmetry at $5.2$ standard deviations, stable under the thirteen variants of the fit and selection and compatible in sign and size with the published analysis of the same data, whose region of the Dalitz plane concentrates it, outside the $\phi(1020)$. It is a raw asymmetry, not a measurement of $\ACP$ and not a significance of CP violation (Sec.~\ref{sec:comparison_published}).""",
"Conclusions: headline (complete)")
t=rd('conclusions.tex')
i=t.index("Each method taught something specific.")
j=t.index("\\end{itemize}",i)+len("\\end{itemize}")
details.append(("Conclusions: lessons of each method (complete)",t[i:j]))
new=r"""Each method taught something specific.
\begin{itemize}[leftmargin=*,itemsep=2pt]
\item \emph{Frequentist.} With the other eleven parameters profiled, Wilks' theorem gives a $68\%$ interval with the nominal coverage ($68.4\pm1.5\%$ in $1000$ bootstrap replicas, $0.70\pm0.01$ and $0.95\pm0.01$ in toy MC, where the estimator is unbiased and reaches the minimum variance bound); Hesse and Minos agree, so the likelihood is parabolic in $A$.
\item \emph{Bayesian.} With flat priors the posterior is the likelihood, and the credible interval coincides with the confidence interval and has the same coverage; the Bayesian analysis adds sampling diagnostics that must be checked (autocorrelation time, dispersed starts), a Bayes factor that depends on the prior width, and a posterior predictive check whose charge statistics, being fitted, have no power.
\item \emph{Machine learning (Appendix~\ref{sec:ml}).} Weak labels from the mass spectrum train classifiers of test AUC $\simeq0.86$; the selection leaves $\Araw$ unchanged, does not improve the precision where the background is small and cannot remove the misidentified decays under the signal.
\end{itemize}"""
t=t[:i]+new+t[j:]
a=t.index("On the systematic study, the dominant contribution"); b=t.index("\n\n",a)
details.append(("Conclusions: systematic study and dependence on choices (complete)",t[a:b]))
t=t[:a]+r"""The dominant systematic contribution is the particle-identification requirement, consistent with a statistical fluctuation of a nested sample; the background model is second and the signal shape does not matter, and the same variants and rule give the same uncertainty in the two methods. Several statistics depend on choices: the $p$-value of a binned $\chi^2$ on the binning, the Bayes factor on the prior, the coverage on whether the toy experiments come from the model or from the data."""+t[b:]
wr('conclusions.tex',t)

out=["\\section{Details Omitted from the Main Text}\n\\label{app:details}\n\\emph{In short: the complete text of the paragraphs that are condensed in the main text, in the order of the sections.}\n"]
for title,txt in details:
    txt=re.sub(r'\\label\{[^}]*\}\n?','',txt)
    out.append("\\subsection{%s}\n%s\n"%(title.replace('_',' '),txt.strip()))
wr('details.tex','\n'.join(out))
a=rd('appendix.tex')
if 'sections/details' not in a:
    wr('appendix.tex',a.replace('\\input{sections/theory}','\\input{sections/theory}\n\\input{sections/details}',1))
print(len(details),'paragraphs saved to details.tex')
