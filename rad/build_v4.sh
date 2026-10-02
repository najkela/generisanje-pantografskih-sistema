set -e
pandoc RAD_v4.md --metadata-file=ieee.yaml -H ieee_header.tex --natbib --bibliography=bib.bib -s -o RAD_v4.tex
python3 - <<'PY'
import re
s=open("RAD_v4.tex").read()
s=s.replace("\\setcounter{secnumdepth}{-\\maxdimen} % remove section numbering","\\setcounter{secnumdepth}{3}")
s=s.replace("\\author{Никола Најић \\and Ментор: Бранислав Милојковић}",
  "\\author{\\IEEEauthorblockN{Никола Најић}\\IEEEauthorblockA{Истраживачка станица Петница\\\\ Ваљево, Србија\\\\ nnajic1@gmail.com, ORCID: N/A}\n\\and\n\\IEEEauthorblockN{Бранислав Милојковић}\\IEEEauthorblockA{Истраживачка станица Петница\\\\ Ваљево, Србија\\\\ branislav.milojkovic@petnica.rs}}")
s=s.replace("\\title{", "\\title{\\makebox[\\textwidth][l]{\\normalfont\\normalsize IEEESTEC -- 19th Student Projects Conference, Ni\\v{s}, 2026.}\\\\[-2pt]\\makebox[\\textwidth][l]{\\normalfont\\footnotesize https://doi.org/ (организатор додаје број)}\\\\[-2pt]\\makebox[\\textwidth][l]{\\normalfont\\footnotesize Original scientific paper}\\\\[6pt] ", 1)
s=s.replace("\\end{abstract}", "\\end{abstract}\n\\begin{IEEEkeywords}\nсинтеза механизама, пантограф, двонивовска оптимизација, генетски алгоритам, ЦМА-ЕС, мешовита оптимизација, Chamfer растојање\n\\end{IEEEkeywords}")
s=s.replace("\\usepackage[]{natbib}","\\usepackage[numbers,sort&compress]{natbib}")
s=s.replace("\\bibliographystyle{plainnat}","\\bibliographystyle{IEEEtran}")
# широка слика преко оба ступца, ужа у ступцу
s=re.sub(r"\\begin\{figure\}\s*\\centering\s*\\includegraphics\[[^\]]*\]\{slike/konvergencija\.png\}",
         "\\\\begin{figure}[t]\n\\\\centering\n\\\\includegraphics[width=\\\\columnwidth]{slike/konvergencija.png}", s)

s=re.sub(r"\\includegraphics\[[^\]]*\]\{slike/parovi\.png\}",
         "\\\\includegraphics[width=0.92\\\\columnwidth]{slike/parovi.png}", s)
s=s.replace("\\bibliography{bib.bib}","\\footnotesize\\bibliography{bib.bib}")
s=s.replace("\\begin{figure}\n\\centering\n\\includegraphics[width=0.84\\columnwidth]{slike/parovi.png}",
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.84\\columnwidth]{slike/parovi.png}")
open("RAD_v4.tex","w").write(s)
PY
xelatex -interaction=nonstopmode RAD_v4.tex >/dev/null 2>&1
bibtex RAD_v4 >/dev/null 2>&1
xelatex -interaction=nonstopmode RAD_v4.tex >/dev/null 2>&1
xelatex -interaction=nonstopmode RAD_v4.tex >/dev/null 2>&1
pdfinfo RAD_v4.pdf | grep Pages
