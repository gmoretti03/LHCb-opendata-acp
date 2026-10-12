"""Regenerates the body/appendix split of the report.

Sources: split_body_appendix/sections_full/*.tex (complete text, every subsection in its natural place) and main_full.tex.
Output: ../sections/*.tex (+ appendix_ext.tex) and ../main.tex. ../sections/frequentist.tex is never touched.
Edit the text in sections_full (except frequentist.tex, edited in ../sections), then run: python3 make_split.py, then compile main.tex.
What goes to the appendix is set in SPEC below (subsections / floats / KEEPL leads / WHOLE sections).
"""
import re, shutil, os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
B=HERE
SRC=HERE+'/sections_full'; DST=os.path.abspath(HERE+'/../sections')
# file -> (appendix title, label, moved subsection prefixes, moved float labels)
SPEC = {
 'data':       ('Dataset and Selection Details','app:ext_data',[],['tab:cutflow']),
 'bayesian':   ('Extended Bayesian Analysis','app:ext_bayes',['Model Adequacy','Robustness Checks','Toy-MC Coverage'],['fig:bayes_sd']),
 'validation': ('Extended Validation of the Estimator','app:ext_valid',['Size and Power','Goodness of Fit'],['fig:validation_injection','fig:validation_null']),
 'dalitz':     ('Extended Dalitz-Plane Analysis','app:ext_dalitz',[],[]),
  'comparison': ('Extended Comparison','app:ext_comp',['Numerical Agreement','Bayesian and Frequentist Intervals at Low'],['tab:comparison']),
}
exec(open(B+'/overrides.py').read()) if os.path.exists(B+'/overrides.py') else None
KEEPL=('Model Adequacy',)
WHOLE=('ml','limitations','code')
FL=re.compile(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}\n?',re.S)
# frequentist.tex is Marco's and lives only in ../sections: never overwritten here
for n in os.listdir(SRC):
    if n=='frequentist.tex': continue
    shutil.copy(f'{SRC}/{n}',f'{DST}/{n}')
shutil.copy(HERE+'/main_full.tex',DST+'/../main.tex')
ext=[]
_app=open(SRC+'/appendix.tex').read()
HYP=[]
for _m in FL.finditer(_app):
    if '\\label{tab:ml_hyperparams}' in _m.group(0) or '\\label{tab:ml_ffnn_search}' in _m.group(0): HYP.append(_m.group(0).rstrip('\n')+'\n')
for f,(title,lab,subs,floats) in SPEC.items():
    t=open(f'{SRC}/{f}.tex').read()
    parts=re.split(r'(?m)^(?=\\subsection\{)',t); keep=[parts[0]]; moved=[]; leads=[]
    for p in parts[1:]:
        m=re.match(r'\\subsection\{(.*)\}',p)
        if m and any(m.group(1).startswith(s) for s in subs):
            le=re.search(r'\\emph\{In short: .*?\}\s*\n',p,re.S); leads.append((m.group(1),le.group(0).strip() if le else '')); moved.append(p.rstrip('\n')+'\n'+(('\n'+'\n'.join(HYP)) if m.group(1).startswith('Hyperparameter') else ''))
        else: keep.append(p)
    body=''.join(keep); fb=[]; fl_cited=[]
    if f=='dalitz':
        m_=re.search(r'\n(Two null tests check the procedure itself.*?)\n\n',body,re.S)
        if m_:
            _L='\\emph{In short: permuting the charges gives $\\chi^2=31.4\\pm8.1$, as expected, and never reaches the observed $80.8$; the phase-space simulation is consistent with no asymmetry, with slightly optimistic $\\chi^2$ $p$-values.}'
            moved.append('\\subsection{Null Tests of the Local-Asymmetry Procedure}\n\\label{sec:dalitz_null}\n'+_L+'\n'+m_.group(1)+'\n'); leads.append(('Null Tests of the Local-Asymmetry Procedure',_L))
            body=body.replace(m_.group(1)+'\n','')
    if f=='ml':
        m_=re.search(r'\n(In the nominal sample the selection cannot improve.*?)\n\n',body,re.S)
        if m_:
            _L='\\emph{In short: with a looser identification ($33{,}328$ candidates, purity $94\\%$) the classifier raises the purity to $99\\%$ and $\\sigma(\\Araw)$ from $0.0072$ to $0.0070$ at best, but cannot remove the misidentified background that moves $\\Araw$ to $-0.055$.}'
            moved.append('\\subsection{High-Background Sample}\n\\label{sec:ml_highbkg}\n'+_L+'\n'+m_.group(1)+'\n'); leads.append(('High-Background Sample',_L))
            body=body.replace(m_.group(1)+'\n','')
    def take(mm):
        for l in floats:
            if f'\\label{{{l}}}' in mm.group(0): fb.append(mm.group(0).rstrip('\n')+'\n'); fl_cited.append(l); return ''
        return mm.group(0)
    body=FL.sub(take,body)
    # citations of moved floats in the body get an explicit pointer to the appendix
    for l in fl_cited:
        body=re.sub(r'((?:Figs?|Tables?)\.?~\\ref\{%s\})'%re.escape(l), r'\1 of Appendix~\\ref{%s}'%lab, body)
    if not (moved or fb): continue
    ptr='\n\\paragraph*{Further material} is in Appendix~\\ref{%s}.'%lab
    items=[]; _titles=[]
    for ti,le in leads:
        le=re.sub(r'^\\emph\{In short:\s*','',le); le=re.sub(r'\}\s*$','',le)
        if ti.startswith(KEEPL): items.append(f' \\emph{{{ti}}}: {le}')
        else: _titles.append(ti)
    body=body.rstrip('\n')+'\n'+ptr+''.join(items)+'\n'
    open(f'{DST}/{f}.tex','w').write(body)
    # floats cited by a moved block go right after that block's citing paragraph
    _keepfb=[];_keepl=[]
    for _l,_f in zip(fl_cited,fb):
        _done=False
        for _i,_blk in enumerate(moved):
            _mm=re.search(r'\\ref\{%s\}'%re.escape(_l),_blk)
            if _mm:
                _end=_blk.find('\n\n',_mm.end())
                _end=len(_blk) if _end<0 else _end
                moved[_i]=_blk[:_end]+'\n\n'+_f+_blk[_end:]
                _done=True;break
        if not _done: _keepfb.append(_f);_keepl.append(_l)
    fb,fl_cited=_keepfb,_keepl
    e=f'\\section{{{title}}}\n\\label{{{lab}}}\n'
    tail=''
    if fb:
        refs=', '.join(('Fig.' if l.startswith('fig') else 'Table')+'~\\ref{%s}'%l for l in fl_cited)
        tail+='\\clearpage\n\\subsection{Float%s Cited in the Main Text}\nThe main text cites %s; %s placed here to keep the main text within its page limit.\n\n'%('s' if len(fb)>1 else '',refs,'they are' if len(fb)>1 else 'it is')+'\n'.join(fb)+'\n'
    e+='\n'.join(moved)+tail+'\n\\FloatBarrier\n'; ext.append((f,e))
order=['data','bayesian','validation','dalitz','comparison']
open(DST+'/appendix_ext.tex','w').write('\n\n'.join(e for k in order for (f,e) in ext if f==k))
m=open(DST+'/../main.tex').read()
for n in WHOLE: m=m.replace('\\input{sections/%s}\n'%n,'')
open(DST+'/../main.tex','w').write(m)
a=open(DST+'/appendix.tex').read()
for _h in HYP: a=a.replace(_h,'')
a=a.replace('\\appendix\n','\\appendix\n\\makeatletter\\renewcommand\\p@subsection{\\thesection.}\\makeatother\n',1)
a=a.replace('\\input{sections/notation}','\\input{sections/appendix_ext}\n'+''.join('\\input{sections/%s}\n'%n for n in WHOLE)+'\\input{sections/notation}')
open(DST+'/appendix.tex','w').write(a)
# vocabulary (not frequentist.tex)
for fn in os.listdir(DST):
    if not fn.endswith('.tex') or fn=='frequentist.tex': continue
    t=open(f'{DST}/{fn}').read(); t0=t
    for a_,b_ in (('Pseudo-experiments','Toy-MC experiments'),('pseudo-experiments','toy-MC experiments'),('Pseudo-experiment','Toy-MC experiment'),('pseudo-experiment','toy-MC experiment'),('pseudo-datasets','toy-MC datasets'),('pseudo-dataset','toy-MC dataset')):
        t=t.replace(a_,b_)
    t=re.sub(r'Sec\.~\\ref\{(sec:limitations|sec:code)\}',r'Appendix~\\ref{\1}',t)
    if t!=t0: open(f'{DST}/{fn}','w').write(t)

def sub_file(fn,pairs):
    t=open(f'{DST}/{fn}').read()
    for a_,b_ in pairs: t=re.sub(a_,b_,t)
    open(f'{DST}/{fn}','w').write(t)
for fn in ('bayesian.tex','dalitz.tex'):
    sub_file(fn,[(r'(Fig\.~\\ref\{[^}]+\})( of Appendix~\\ref\{[^}]+\})([a-z])',r'\1\3\2')])

extt=open(DST+'/appendix_ext.tex').read()
lab2app={}
for sec in re.split(r'(?m)^(?=\\section\{)',extt):
    ml=re.search(r'\\label\{(app:[^}]+)\}',sec)
    if ml:
        for l in re.findall(r'\\label\{((?:fig|tab):[^}]+)\}',sec): lab2app[l]=ml.group(1)
for fn in ['introduction','data','model','bayesian','systematics','dalitz','ml','comparison','conclusions','abstract','details']:
    t=open(f'{DST}/{fn}.tex').read(); t0=t
    for l,ap in lab2app.items():
        t=re.sub(r'((?:Figs?\.|Figure|Tables?)~\\ref\{%s\})(?![a-z]? of Appendix)'%re.escape(l),r'\1 of Appendix~\\ref{%s}'%ap,t)
    if t!=t0: open(f'{DST}/{fn}.tex','w').write(t)

_ext=open(DST+'/appendix_ext.tex').read()
_applab=set(re.findall(r'\\label\{(sec:[^}]+)\}',_ext))|set(re.findall(r'\\label\{(sec:[^}]+)\}',open(DST+'/ml.tex').read()))
_applab-={'sec:ml'}
for fn in os.listdir(DST):
    if not fn.endswith('.tex') or fn in ('frequentist.tex','appendix_ext.tex','ml.tex','theory.tex','appendix_checks.tex','appendix.tex','details.tex','limitations.tex','code.tex','notation.tex'): continue
    t=open(f'{DST}/{fn}').read(); t0=t
    t=re.sub(r'(\\ref\{[^}]+\}) of Appendix~\\ref\{([^}]+)\}([a-z])\b',r'\1\3 of Appendix~\\ref{\2}',t)
    for l in _applab:
        t=re.sub(r'Secs?\.~\\ref\{%s\}'%re.escape(l),lambda _m,l=l:'Appendix~\\ref{'+l+'}',t)
    if t!=t0: open(f'{DST}/{fn}','w').write(t)
print('ok',[f for f,_ in ext])
