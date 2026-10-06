set -e
pandoc RAD_v5.md --metadata-file=ieee_v5.yaml -H ieee_header_v5.tex --natbib --bibliography=bib.bib -s -o RAD_v5.tex
python3 - <<'PY'
import re
s=open("RAD_v5.tex").read()
s=s.replace("\\setcounter{secnumdepth}{-\\maxdimen} % remove section numbering","\\setcounter{secnumdepth}{3}")
s=s.replace("\\author{Nikola Najić \\and Mentor: Branislav Milojković}",
  "\\author{\\IEEEauthorblockN{Nikola Najić}\\IEEEauthorblockA{Istraživačka stanica Petnica\\\\ Valjevo, Srbija\\\\ nnajic1@gmail.com, ORCID: N/A}\n\\and\n\\IEEEauthorblockN{Branislav Milojković}\\IEEEauthorblockA{Istraživačka stanica Petnica\\\\ Valjevo, Srbija\\\\ branislav.milojkovic@petnica.rs}}")
s=s.replace("\\title{", "\\title{\\makebox[\\textwidth][l]{\\normalfont\\normalsize IEEESTEC -- Student Projects Conference, Ni\\v{s}, 2026.}\\\\[-2pt]\\makebox[\\textwidth][l]{\\normalfont\\footnotesize https://doi.org/ (organizator dodaje broj)}\\\\[-2pt]\\makebox[\\textwidth][l]{\\normalfont\\footnotesize Original scientific paper}\\\\[6pt] ", 1)
s=s.replace("\\end{abstract}", "\\end{abstract}\n\\begin{IEEEkeywords}\nsinteza mehanizama, pantografski mehanizam, dvonivovska optimizacija, genetski algoritam, CMA-ES, mešovita optimizacija, Chamfer rastojanje\n\\end{IEEEkeywords}")
s=s.replace("\\usepackage[]{natbib}","\\usepackage[numbers,sort&compress]{natbib}")
s=s.replace("\\bibliographystyle{plainnat}","\\bibliographystyle{IEEEtran}")
# široka slika preko oba stupca, uža u stupcu
s=re.sub(r"\\begin\{figure\}\s*\\centering\s*\\includegraphics\[[^\]]*\]\{slike/konvergencija_v5\.png\}",
         "\\\\begin{figure}[t]\n\\\\centering\n\\\\includegraphics[width=\\\\columnwidth]{slike/konvergencija_v5.png}", s)

s=re.sub(r"\\includegraphics\[[^\]]*\]\{slike/parovi\.png\}",
         "\\\\includegraphics[width=0.92\\\\columnwidth]{slike/parovi.png}", s)
s=s.replace("\\bibliography{bib.bib}","\\scriptsize\\bibliography{bib.bib}")
s=s.replace("\\begin{figure}\n\\centering\n\\includegraphics[width=0.84\\columnwidth]{slike/parovi.png}",
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.84\\columnwidth]{slike/parovi.png}")
open("RAD_v5.tex","w").write(s)
PY
xelatex -interaction=nonstopmode RAD_v5.tex >/dev/null 2>&1
bibtex RAD_v5 >/dev/null 2>&1
xelatex -interaction=nonstopmode RAD_v5.tex >/dev/null 2>&1
xelatex -interaction=nonstopmode RAD_v5.tex >/dev/null 2>&1
pdfinfo RAD_v5.pdf | grep Pages
