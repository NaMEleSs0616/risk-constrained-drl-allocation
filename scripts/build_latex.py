#!/usr/bin/env python
"""Convert the built HTML paper into a self-contained LaTeX submission folder.

Output: ``paper/latex/main.tex`` plus the figures it uses under ``paper/latex/figures``.
The source compiles with pdfLaTeX (all Unicode symbols are mapped to LaTeX), so the
folder can be uploaded to Overleaf / arXiv as is. Sections are numbered by LaTeX, and
citations become ``\\cite`` keys into a ``thebibliography`` built from the reference list.

Usage:
    python scripts/build_latex.py --html paper/paper.html --out paper/latex
"""

from __future__ import annotations

import argparse
import html as htmllib
import re
import shutil
import sys
from pathlib import Path
from typing import Sequence

import pypandoc

UNICODE_TO_LATEX = {
    "λ₁": r"$\lambda_1$", "λ₂": r"$\lambda_2$", "λ₃": r"$\lambda_3$", "λ": r"$\lambda$",
    "±": r"$\pm$", "−": r"$-$", "×": r"$\times$", "≤": r"$\le$", "≥": r"$\ge$", "→": r"$\to$",
    "Σ": r"$\Sigma$", "·": r"$\cdot$", "⁻⁴": r"$^{-4}$", "ν": r"$\nu$", "ρ": r"$\rho$", "η": r"$\eta$",
    "γ": r"$\gamma$", "κ": r"$\kappa$", "α": r"$\alpha$", "σ": r"$\sigma$", "Φ": r"$\Phi$", "Δ": r"$\Delta$",
    "₁": r"$_1$", "₂": r"$_2$", "₃": r"$_3$", "²": r"$^2$", "³": r"$^3$", "≈": r"$\approx$", "…": r"\ldots{}",
    "–": "--", "—": "---", "§": r"\S{}", "’": "'", "‘": "`", "“": "``", "”": "''",
}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,longtable,array,calc,adjustbox}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{microtype}
\usepackage[section]{placeins}
\makeatletter\def\fps@figure{!htbp}\makeatother
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\makeatletter
\newsavebox\pandoc@box
\newcommand*\pandocbounded[1]{%
  \sbox\pandoc@box{#1}%
  \Gscale@div\@tempa{\textheight}{\dimexpr\ht\pandoc@box+\dp\pandoc@box\relax}%
  \Gscale@div\@tempb{\linewidth}{\wd\pandoc@box}%
  \ifdim\@tempb\p@<\@tempa\p@\let\@tempa\@tempb\fi
  \ifdim\@tempa\p@<\p@\scalebox{\@tempa}{\usebox\pandoc@box}%
  \else\usebox{\pandoc@box}\fi}
\makeatother
\setlength{\LTcapwidth}{\textwidth}
\graphicspath{{./}}

\title{%TITLE%}
\author{%AUTHOR%}
\date{October 2026}

\begin{document}
\maketitle
\begin{abstract}
%ABSTRACT%
\end{abstract}
\noindent\textbf{Keywords:} %KEYWORDS%

"""


def to_latex(fragment: str) -> str:
    """Convert an HTML fragment to LaTeX with pandoc."""
    return pypandoc.convert_text(fragment, "latex", format="html+tex_math_dollars",
                                 extra_args=["--wrap=preserve", "--shift-heading-level-by=-1"])


def preprocess(main: str) -> tuple[str, str, str, str, str, list[str]]:
    """Pull out title / abstract / keywords and normalise the body for pandoc."""
    title = re.search(r"<h1>(.*?)</h1>", main, re.S).group(1)
    author_m = re.search(r'<div class="author">(.*?)</div>', main, re.S)
    author = author_m.group(1).strip() if author_m else "Anonymous"
    abstract = re.search(r'<section class="abstract">.*?<h2>Abstract</h2>(.*?)</section>', main, re.S).group(1)
    keywords = re.search(r'<p class="keywords"><b>Keywords:</b>(.*?)</p>', main, re.S).group(1).strip()
    body = main[main.index('<h2 id="s1">'):]
    figures = re.findall(r'<img src="([^"]+)"', body)

    body = re.sub(r'<span class="sec">[^<]*</span>', "", body)                  # LaTeX numbers sections
    body = re.sub(r"<b>(Table|Figure) \d+\.</b>\s*", "", body)                    # LaTeX numbers floats
    # Tables: move the <figcaption> into the <table> as a <caption>; keep notes after the table.
    def table_fig(m: re.Match) -> str:
        block = m.group(0)
        tid = re.match(r"<figure class=['\"]table['\"][^>]*?id=['\"]([^'\"]+)", block)
        cap = re.search(r"<figcaption>(.*?)</figcaption>", block, re.S)
        caption = cap.group(1).strip() if cap else ""
        table = re.search(r"<table>.*?</table>", block, re.S).group(0)
        if tid:
            caption += f" LBLK{tid.group(1)}LBLE"
        table = table.replace("<table>", f"<table><caption>{caption}</caption>", 1)
        notes = " ".join(re.findall(r"<p class=['\"]table-note['\"]>(.*?)</p>", block, re.S))
        if notes:  # keep notes with their table: append to the caption
            table = table.replace("</caption>", f" <i>Note:</i> {notes}</caption>", 1)
        return table
    body = re.sub(r"<figure class=['\"]table['\"].*?</figure>", table_fig, body, flags=re.S)
    # Citations -> placeholders that survive pandoc; merged later into \cite{...}.
    body = re.sub(r'<a href="#(r\d+)">\[\d+\]</a>', r"CITEKEY\1CITEEND", body)
    # Internal cross-reference links -> plain text (numbers already match LaTeX's order).
    body = re.sub(r"<a href=['\"]#((?:tab|fig)-[^'\"]+)['\"]>(Table|Figure) \d+</a>", r"\2 REFK\1REFE", body)
    body = re.sub(r"<a href=['\"]#(?:tab|fig)-[^'\"]+['\"]>(.*?)</a>", r"\1", body)
    body = re.sub(r'<a href="#(s\d+|appendix|refs)">(.*?)</a>', r"\2", body)
    # Math delimiters.
    body = re.sub(r"\\\[(.*?)\\\]", lambda m: "$$" + m.group(1) + "$$", body, flags=re.S)
    body = re.sub(r"\\\((.*?)\\\)", lambda m: "$" + m.group(1) + "$", body, flags=re.S)
    # Findings / callouts -> plain paragraphs.
    body = re.sub(r'<div class="finding"><span class="n">(R\d)</span><p>', r"<p><b>\1.</b> ", body)
    body = body.replace('<div class="callout">', "<blockquote>").replace("</div>", "</blockquote>") \
        if False else re.sub(r'<div class="callout">(.*?)</div>', r"<blockquote>\1</blockquote>", body, flags=re.S)
    body = re.sub(r"</p></div>", "</p>", body)
    return title, author, abstract, keywords, body, figures


def references(body: str) -> tuple[str, str]:
    """Split off the reference list and render it as thebibliography."""
    m = re.search(r'<h2 id="refs">References</h2>\s*<ol class="refs">(.*?)</ol>', body, re.S)
    items = re.findall(r'<li id="(r\d+)">(.*?)</li>', m.group(1), re.S)
    bib = ["\\begin{thebibliography}{99}"]
    for key, text in items:
        bib.append(f"\\bibitem{{{key}}} " + to_latex(text).strip())
    bib.append("\\end{thebibliography}")
    return body[:m.start()] + body[m.end():], "\n".join(bib)


def longtable_to_float(m: re.Match) -> str:
    """Rewrite a pandoc longtable as a table float scaled to the text width."""
    block = m.group(0)
    spec = re.match(r"\\begin\{longtable\}\[\]\{(.*?)\}\n", block).group(1)
    cap = re.search(r"\\caption\{(.*?)\}\\tabularnewline", block, re.S)
    head = re.search(r"\\toprule\\noalign\{\}\n(.*?)\\midrule\\noalign\{\}\n\\endfirsthead", block, re.S)
    rows = re.search(r"\\endlastfoot\n(.*?)\\end\{longtable\}", block, re.S)
    caption = cap.group(1) if cap else ""
    return ("\\begin{table}[!htbp]\n\\centering\\scriptsize\n"
            f"\\caption{{{caption}}}\n\\begin{{adjustbox}}{{max width=\\linewidth}}\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head.group(1) if head else ''}\\midrule\n"
            f"{rows.group(1).rstrip() if rows else ''}\n\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n"
            "\\end{table}")


def postprocess(tex: str) -> str:
    tex = re.sub(r"\s*LBLK([\w-]+)LBLE", r"\\label{\1}", tex)
    tex = re.sub(r"(Table|Figure) REFK([\w-]+)REFE", r"\1~\\ref{\2}", tex)
    tex = re.sub(r"(?:CITEKEY(r\d+)CITEEND)+",
                 lambda m: "\\cite{" + ",".join(re.findall(r"CITEKEY(r\d+)CITEEND", m.group(0))) + "}", tex)
    tex = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", longtable_to_float, tex, flags=re.S)
    # Figures inside \pandocbounded are scaled to the line width.
    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{", r"\\includegraphics[width=\\linewidth]{", tex)
    tex = tex.replace("\\begin{verbatim}", "{\\scriptsize\\begin{verbatim}").replace("\\end{verbatim}", "\\end{verbatim}}")
    tex = re.sub(r"\\texttt\{(github\.com/[^}]+)\}", r"\\url{https://\1}", tex)
    for k in sorted(UNICODE_TO_LATEX, key=len, reverse=True):
        tex = tex.replace(k, UNICODE_TO_LATEX[k])
    tex = tex.replace("$$-$", "$-")  # collapse doubled math shifts produced by minus signs in math
    return tex


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html", default="paper/paper.html")
    p.add_argument("--out", default="paper/latex")
    args = p.parse_args(argv)
    src = Path(args.html)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    page = src.read_text(encoding="utf-8")
    main_html = re.search(r'<main class="page">(.*)</main>', page, re.S).group(1)
    title, author, abstract, keywords, body, figures = preprocess(main_html)
    body, bib = references(body)
    appendix_split = body.split('<h2 id="appendix">Appendix</h2>')
    tex_body = to_latex(appendix_split[0])
    if len(appendix_split) > 1:
        tex_body += "\n\\clearpage\n" + bib + "\n\n\\appendix\n\\section*{Appendix}\n" + to_latex(appendix_split[1])
    else:
        tex_body += "\n" + bib
    doc = (PREAMBLE.replace("%TITLE%", to_latex(title).strip())
           .replace("%AUTHOR%", htmllib.unescape(author))
           .replace("%ABSTRACT%", to_latex(abstract).strip())
           .replace("%KEYWORDS%", htmllib.unescape(keywords))
           + tex_body + "\n\\end{document}\n")
    doc = postprocess(doc)
    (out / "main.tex").write_text(doc, encoding="utf-8")
    for fig in figures:
        dst = out / fig
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src.parent / fig, dst)

    # Structural checks (no TeX engine is assumed to be installed).
    problems = []
    for env in ("table", "tabular", "adjustbox", "figure", "abstract", "thebibliography", "document"):
        b, e = doc.count(f"\\begin{{{env}}}"), doc.count(f"\\end{{{env}}}")
        if b != e:
            problems.append(f"unbalanced {env}: {b} begin / {e} end")
    non_ascii = sorted({ch for ch in doc if ord(ch) > 127 and not ch.isalpha()})
    if non_ascii:
        problems.append(f"unmapped non-ASCII symbols: {non_ascii}")
    if doc.count("{") != doc.count("}"):
        problems.append(f"brace imbalance: {doc.count('{')} vs {doc.count('}')}")
    print(f"wrote {out / 'main.tex'} ({len(doc):,} chars), {len(figures)} figures, "
          f"{doc.count(chr(92) + 'cite{')} citations, {doc.count(chr(92) + 'section{')} sections")
    for prob in problems:
        print("CHECK:", prob, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
