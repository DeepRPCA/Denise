import os
import tensorflow as tf


_PDF_TMPL = r"""
\documentclass[3p, twocolumn]{elsarticle}
\usepackage[colorlinks, citecolor=blue]{hyperref}
\usepackage{graphicx}
\usepackage{subfigure}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{tabularx}
\usepackage{enumerate}
\newcommand{\folder}{N20_K3_fr3_S095/0/}
\newcommand{\foldermarket}{N20_market/0/}
\usepackage{wrapfig}
\usepackage{algorithm,algorithmic}
\usepackage{xparse}
\usepackage{numcompress}
\begin{document}
%(tables)s
\end{document}
"""


if __name__ == '__main__':

    inputs = ""
    for i, fn in enumerate(sorted(os.listdir("../comparisons/"))):
        inputs += "\n\n{}: {}\n".format(i+1, fn[:-4].replace("_", " "))
        inputs += "\input{../code_src/comparisons/" + "{}".format(fn[:-4]) + "}"
        if (i+1) % 10 == 0:
            inputs += "\clearpage \n"

    print(inputs)

    pdf_tex = _PDF_TMPL % {"tables": inputs}

    with open("../../latex_src/Tables.tex", "w") as tex:
        tex.write(pdf_tex)


