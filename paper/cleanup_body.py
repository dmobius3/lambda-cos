"""
Step 2 cleanup: body-raw.tex (raw pandoc output) -> body.tex (compileable body).

Order matters: text-symbol replacements that introduce '$' but no '{}' run
BEFORE heading-level bumping (whose regex stops at the first '}'), and
text-mode underscore -> math-mode replacements (which introduce '{}') run
AFTER heading bumping.

Operations (in order):
  1. Strip repo-nav links + horizontal rules at top and bottom.
  2. Remove duplicate \\section{Title} (already in \\title{}).
  3. Remove \\subsection{Abstract} block (already in \\begin{abstract}).
  4. \\textgreater{}/\\textless{} -> $>$/$<$
  5. Bump heading levels and strip numeric prefixes + auto-labels.
  6. Text-mode subscripts -> math mode.
  7. Collapse runs of 3+ blank lines.

Defer to step 3+: numeric citations, figure environments, cross-refs.
"""

import re
from pathlib import Path

src = Path("body-raw.tex").read_text(encoding="utf-8")

# --- 1: strip nav-link block(s) and horizontal-rule lines ---
def is_nav_or_hr(line):
    return ("\\href{../" in line) or ("\\rule{0.5\\linewidth}" in line)

lines = src.splitlines()
i = 0
while i < len(lines) and (is_nav_or_hr(lines[i]) or lines[i].strip() == ""):
    i += 1
lines = lines[i:]
while lines and (is_nav_or_hr(lines[-1]) or lines[-1].strip() == ""):
    lines.pop()
lines = [ln for ln in lines if "\\rule{0.5\\linewidth}" not in ln]
src = "\n".join(lines) + "\n"

# --- 2: remove the duplicate \section{Apparent Phantom Crossing...} block ---
src = re.sub(
    r"\\section\{Apparent Phantom Crossing[^}]*\n[^}]*\}\\label\{[^}]+\}\n+",
    "",
    src,
    count=1,
    flags=re.DOTALL,
)
# Author/email line that follows the title in the markdown.
src = re.sub(
    r"\\textbf\{B\. Shatto\}[^\n]*\n[^\n]*\n[^\n]*\\href\{mailto[^\n]+\n+",
    "",
    src,
    count=1,
)

# --- 3: remove the \subsection{Abstract}...content block ---
src = re.sub(
    r"\\subsection\{Abstract\}\\label\{abstract\}.*?(?=\\subsection\{1\.)",
    "",
    src,
    flags=re.DOTALL,
)

# --- 4: textgreater/textless -> $>$/$<$ (no '{}' introduced; safe before headings) ---
src = src.replace(r"\textgreater{}", r"$>$")
src = src.replace(r"\textless{}",   r"$<$")

# --- 5: bump heading levels + strip numeric prefixes + drop auto-labels ---
# Now that step 4 replaced \textgreater{}, no headings contain '{}' in their titles.
# \subsubsection{N.M Title}\label{...} -> \subsection{Title}
src = re.sub(
    r"\\subsubsection\{(?:\d+\.\d+\s+)?([^}]+)\}\\label\{[^}]+\}",
    r"\\subsection{\1}",
    src,
)
# \subsection{N. Title}\label{...} -> \section{Title}
src = re.sub(
    r"\\subsection\{(?:\d+\.\s+)?([^}]+)\}\\label\{[^}]+\}",
    r"\\section{\1}",
    src,
)
# Defensive: drop any straggler \label{...} that survived.
src = re.sub(r"\}\\label\{[^}]+\}", "}", src)

# --- 6: text-mode subscripts -> math mode (introduces '{}'; runs AFTER headings) ---
subs = [
    (r"D\_H",   r"$D_H$"),
    (r"D\_M",   r"$D_M$"),
    (r"D\_V",   r"$D_V$"),
    (r"M\_B",   r"$M_B$"),
    (r"l\_A",   r"$\ell_A$"),
    (r"r\_d",   r"$r_d$"),
    (r"t\_now", r"$t_{\rm now}$"),
    (r"w\_a",   r"$w_a$"),
    (r"w\_eff", r"$w_{\rm eff}$"),
    (r"Ω\_k",   r"$\Omega_k$"),
    (r"Ω\_m",   r"$\Omega_m$"),
    (r"Ω\_r",   r"$\Omega_r$"),
    (r"Ω\_Λ",   r"$\Omega_\Lambda$"),
]
for pat, repl in subs:
    src = src.replace(pat, repl)

# --- 6b: unicode mathematical characters -> LaTeX math mode ---
# pdflatex without lualatex/xelatex needs these escaped. Each char wraps
# in $...$ standalone; adjacent runs merge visually (e.g., "α(1+z)³"
# becomes "$\alpha$(1+z)$^3$" — spacing is slightly off but readable).
# Polish: future pass can wrap whole expressions in single $...$.
unicode_math = {
    # Unicode minus -> ASCII minus (no math wrap needed; works in text & math)
    "−": "-",
    # Plus/minus and other relations
    "≈": r"$\approx$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "≠": r"$\neq$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "→": r"$\to$",
    "∞": r"$\infty$",
    "∝": r"$\propto$",
    "∫": r"$\int$",
    "∂": r"$\partial$",
    "⊗": r"$\otimes$",
    "∈": r"$\in$",
    "′": r"'",
    "°": r"$^\circ$",
    # Mathcal K used in body
    "𝒦": r"$\mathcal{K}$",
    # Greek lowercase
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "ε": r"$\varepsilon$",
    "ζ": r"$\zeta$",
    "η": r"$\eta$",
    "θ": r"$\theta$",
    "ι": r"$\iota$",
    "κ": r"$\kappa$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "ν": r"$\nu$",
    "ξ": r"$\xi$",
    "π": r"$\pi$",
    "ρ": r"$\rho$",
    "σ": r"$\sigma$",
    "τ": r"$\tau$",
    "υ": r"$\upsilon$",
    "φ": r"$\varphi$",
    "χ": r"$\chi$",
    "ψ": r"$\psi$",
    "ω": r"$\omega$",
    # Greek uppercase that differ from Latin
    "Γ": r"$\Gamma$",
    "Δ": r"$\Delta$",
    "Θ": r"$\Theta$",
    "Λ": r"$\Lambda$",
    "Ξ": r"$\Xi$",
    "Π": r"$\Pi$",
    "Σ": r"$\Sigma$",
    "Υ": r"$\Upsilon$",
    "Φ": r"$\Phi$",
    "Ψ": r"$\Psi$",
    "Ω": r"$\Omega$",
    # (Numeric/letter super/sub-scripts handled below as runs, not single chars,
    # so adjacent ones like ⁻³ -> $^{-3}$ instead of $^-$$^3$ which produces
    # the LaTeX "double superscript" error.)
    # Misc symbols
    "§": r"\S",
    "†": r"$^\dagger$",
    "≳": r"$\gtrsim$",
    "≡": r"$\equiv$",
    "≪": r"$\ll$",
    "√": r"$\surd$",
    "½": r"$1/2$",
    # Latin diacritics (e.g., Cortês)
    "ê": r"\^{e}",
}
for u, latex in unicode_math.items():
    src = src.replace(u, latex)

# Super/sub script *runs* -> single math expression.
# Handles ⁻³ -> $^{-3}$ rather than $^-$$^3$ which would produce a
# "double superscript" error after the $$-collapse step.
SUPER_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "ⁿ": "n", "ᵀ": "T",
}
SUB_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ᵢ": "i",
}

def _super_to_math(m):
    s = "".join(SUPER_MAP[c] for c in m.group(0))
    return "$^{" + s + "}$" if len(s) > 1 else "$^" + s + "$"

def _sub_to_math(m):
    s = "".join(SUB_MAP[c] for c in m.group(0))
    return "$_{" + s + "}$" if len(s) > 1 else "$_" + s + "$"

src = re.sub("[" + "".join(SUPER_MAP.keys()) + "]+", _super_to_math, src)
src = re.sub("[" + "".join(SUB_MAP.keys()) + "]+", _sub_to_math, src)

# Adjacent $...$ $...$ runs collapse to a single $...$ for cleaner output:
# e.g., "$\alpha$(1+z)$^3$" stays as is, but "$_0$$^2$" -> "$_0^2$".
# Be conservative: only collapse direct $$ neighbors.
src = re.sub(r"\$\$", "", src)

# --- 6c: simplify pandoc's complex longtable column specs ---
# Pandoc emits things like:
#   \begin{longtable}[]{@{}
#     >{\raggedright\arraybackslash}p{(\linewidth - 6\tabcolsep) * \real{0.2982}}
#     >{\raggedright\arraybackslash}p{(\linewidth - 6\tabcolsep) * \real{0.3509}}
#     @{}}
# which fights with revtex (\real, calc, * length-multiplication, etc.).
# Replace with plain left-aligned columns of the same count: @{}lll...@{}.
def _fix_complex_longtable(m):
    n_cols = m.group(0).count(r"\raggedright\arraybackslash")
    return r"\begin{longtable}[]{@{}" + "l" * n_cols + r"@{}}"

src = re.sub(
    r"\\begin\{longtable\}\[\]\{@\{\}\s*"
    r"(?:>\{\\raggedright\\arraybackslash\}p\{[\s\S]*?\}\}\s*)+"
    r"@\{\}\}",
    _fix_complex_longtable,
    src,
)

# --- 6c.2: strip pandoc's \begin{minipage}[b]{\linewidth} cell wrappers ---
# These wrappers force each cell to \linewidth (the full column), which
# overflows in two-column layout. Removing them lets the table autosize.
# Permissive whitespace pattern catches both multi-line cells and empty
# cells (e.g., the leading empty column in the §5.2 primary-fit table).
src = re.sub(
    r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\s*([\s\S]*?)\s*\\end\{minipage\}",
    lambda m: m.group(1).strip(),
    src,
)

# --- 6c.3: convert longtable -> tabular inside table or table* ---
# Pandoc emits longtable for everything, but our tables fit on one page so
# tabular is fine and gives more layout flexibility. Wide tables (6+ cols)
# span both columns via table*; narrower tables stay column-wide. All get
# \footnotesize for tighter fit in two-column.
def _longtable_to_tabular(m):
    block = m.group(0)
    # Column spec uses @{}<cols>@{}; capture <cols> non-greedily.
    spec_m = re.search(
        r"\\begin\{longtable\}\[\]\{@\{\}([\s\S]*?)@\{\}\}",
        block,
    )
    cols = spec_m.group(1) if spec_m else ""
    n_cols = len(cols)
    # Span both columns whenever there are 3+ columns. Two-column tables
    # (e.g., the FLRW component / redshift-scaling table) stay column-wide.
    # NOTE: revtex+array trip on p{...} columns ("Extra \\or" errors), so
    # we keep plain `l` columns. A few wide-content tables overflow by
    # 100-260 pt; flagged for manual polish (use \resizebox or rewrite).
    env = "table*" if n_cols >= 3 else "table"
    col_spec = f"@{{}}{cols}@{{}}"
    tab_open = f"\\begin{{tabular}}{{{col_spec}}}"
    tab_close = "\\end{tabular}"

    # Cut everything from \begin{longtable}[]{@{}...@{}} (inclusive)
    # through \end{longtable} (exclusive).
    inner = re.sub(
        r"^\\begin\{longtable\}\[\]\{@\{\}[\s\S]*?@\{\}\}\s*",
        "",
        block,
    )
    inner = re.sub(r"\s*\\end\{longtable\}\s*$", "", inner)

    # Strip longtable-specific markers.
    inner = re.sub(r"\\noalign\{\}", "", inner)
    # Drop the longtable footer block: \bottomrule \endlastfoot (in either order
    # with optional \endhead between).
    inner = re.sub(
        r"\\endhead\s*\\bottomrule\s*\\endlastfoot",
        "",
        inner,
    )
    # Defensive: remove any remaining \endhead / \endlastfoot fragments.
    inner = re.sub(r"\\endhead\s*", "", inner)
    inner = re.sub(r"\\endlastfoot\s*", "", inner)

    return (
        f"\\begin{{{env}}}[!htb]\n"
        "\\centering\\footnotesize\n"
        f"{tab_open}\n"
        + inner.strip()
        + "\n\\bottomrule\n"
        f"{tab_close}\n"
        f"\\end{{{env}}}"
    )

src = re.sub(
    r"\\begin\{longtable\}[\s\S]*?\\end\{longtable\}",
    _longtable_to_tabular,
    src,
)

# --- 6d: numeric citations -> \cite{refN} ---
# Pandoc rendered "[N]" as "{[}N{]}" with optional comma-separated lists like
# "{[}4,5{]}" (no space after comma — distinguishes from credible intervals
# like "{[}9977, 10111{]}" which have a space and are left alone).
def _to_cite(m):
    nums = m.group(1).split(",")
    return r"\cite{" + ",".join(f"ref{n}" for n in nums) + "}"

src = re.sub(r"\{\[\}(\d{1,2}(?:,\d{1,2})*)\{\]\}", _to_cite, src)

# --- 6e: drop the markdown references section (replaced by \bibliography) ---
src = re.sub(r"\\section\{References\}.*$", "", src, flags=re.DOTALL)

# --- 6f: figure callouts -> \begin{figure} environments ---
# Markdown:  [Figure N: caption text spanning lines]
# Pandoc:    {[}Figure N: caption text spanning lines{]}
# LaTeX:     \begin{figure}[h] \includegraphics... \caption{...} \label{fig:N} \end{figure}
FIG_FILES = {
    "1": "fig1_template_bias_overlay.pdf",
    "2": "fig2_threshold_scan.pdf",
    "3": "fig3_lcos_corner.pdf",
    "4": "fig4_hubble_residuals.pdf",
}

def _to_figure(m):
    n = m.group(1)
    caption = re.sub(r"\s+", " ", m.group(2).strip())
    f = FIG_FILES.get(n, f"fig{n}.pdf")
    return (
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\columnwidth]{{figures/{f}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{n}}}\n"
        "\\end{figure}"
    )

src = re.sub(
    r"\{\[\}Figure\s+(\d+):\s*([\s\S]*?)\{\]\}",
    _to_figure,
    src,
)

# --- 6g: section labels via brace-matched title -> label lookup ---
# Multi-line headings (e.g., \section{Template Bias\nDemonstration}) are
# normalized to single-line form during this pass.
SECTION_LABELS = {
    "Introduction":                          "sec:intro",
    "The $\\Lambda$cos Model":               "sec:lcos-model",
    "Background ansatz":                     "sec:ansatz",
    "Clock selection":                       "sec:clock",
    "The Hubble rate":                       "sec:hubble",
    "The (1+z)$^1$ term":                    "sec:1z-term",
    "$w_{\\rm eff}$(z) $>$ -1 at All Redshifts": "sec:weff-proof",
    "Template Bias Demonstration":           "sec:template-bias",
    "Method":                                "sec:tb-method",
    "Results":                               "sec:tb-results",
    "Threshold scan":                        "sec:threshold-scan",
    "Comparison with the DESI best fit":     "sec:desi-comparison",
    "Observational Constraints":             "sec:obs-constraints",
    "Data":                                  "sec:data",
    "Primary fit: SN+BAO":                   "sec:primary-fit",
    "Prior sensitivity":                     "sec:prior-sensitivity",
    "$\\Omega_\\Lambda$ sensitivity":        "sec:omega-sensitivity",
    "CMB distance priors":                   "sec:cmb-priors",
    "Parameter accounting":                  "sec:param-accounting",
    "Model comparison":                      "sec:model-comparison",
    "Predictions and Falsification Criteria": "sec:predictions",
    "The (1+z)$^1$ signature":               "sec:1z-signature",
    "Falsification criteria":                "sec:falsification",
    "Discussion":                            "sec:discussion",
    "Template bias":                         "sec:disc-tb",
    "Status of $\\Lambda$cos":               "sec:disc-status",
    "Conclusions":                           "sec:conclusions",
    "Data and Code Availability":            "sec:data-availability",
    "Appendix A: Clock Exponent Selection":  "sec:appendix",
}

def _add_section_labels(s):
    out = []
    pos = 0
    for m in re.finditer(r"\\(section|subsection)\{", s):
        out.append(s[pos:m.start()])
        cmd = m.group(1)
        j = m.end()
        # brace-balanced search for the matching }
        depth, k = 1, j
        while k < len(s) and depth > 0:
            if s[k] == "{": depth += 1
            elif s[k] == "}": depth -= 1
            k += 1
        title_raw = s[j:k-1]
        title_norm = re.sub(r"\s+", " ", title_raw).strip()
        label = SECTION_LABELS.get(title_norm)
        heading = f"\\{cmd}{{{title_norm}}}"
        if label:
            heading += f"\\label{{{label}}}"
        out.append(heading)
        pos = k
    out.append(s[pos:])
    return "".join(out)

src = _add_section_labels(src)

# --- 6h: cross-references \S<N.M> -> Sec.~\ref{sec:...} ---
# Sorted by key length descending so \S5.7 is replaced before \S5.
SREF_MAP = {
    "\\S2.1": "Sec.~\\ref{sec:ansatz}",
    "\\S2.2": "Sec.~\\ref{sec:clock}",
    "\\S2.3": "Sec.~\\ref{sec:hubble}",
    "\\S2.4": "Sec.~\\ref{sec:1z-term}",
    "\\S3":   "Sec.~\\ref{sec:weff-proof}",
    "\\S4.3": "Sec.~\\ref{sec:threshold-scan}",
    "\\S4":   "Sec.~\\ref{sec:template-bias}",
    "\\S5.2": "Sec.~\\ref{sec:primary-fit}",
    "\\S5.4": "Sec.~\\ref{sec:omega-sensitivity}",
    "\\S5.5": "Sec.~\\ref{sec:cmb-priors}",
    "\\S5.7": "Sec.~\\ref{sec:model-comparison}",
    "\\S5":   "Sec.~\\ref{sec:obs-constraints}",
}
for old in sorted(SREF_MAP, key=lambda k: -len(k)):
    src = src.replace(old, SREF_MAP[old])

# --- 6i: text-mode subscript labels in math expressions ---
# "$<math>$\_<word>" -> "$<math>_{\rm <word>}$"  (the upright \rm subscript
# is the convention for short text labels: SN, BAO, min, max, eff, etc.)
src = re.sub(
    r"\$([^\$]+)\$\\_([A-Za-z][A-Za-z0-9]*)",
    r"$\1_{\\rm \2}$",
    src,
)

# --- 6j: specific math fixes the auto-pass couldn't infer ---
# "S^{-3}/$^2$" should be "S^{-3/2}" (auto-pass split unicode ⁻³/² into
# two separate superscripts; merge them back into a single fraction power).
src = src.replace(r"S$^{-3}$/$^2$", r"$S^{-3/2}$")
src = src.replace(r"$\mathcal{K}^1$/$^2$", r"$\mathcal{K}^{1/2}$")

# "H$_0r_d$" - the H is in text, the rest in math; rendering is awkward.
# Wrap the whole product as one math expression.
src = src.replace(r"H$_0r_d$", r"$H_0 r_d$")

# --- 7: collapse 3+ blank-line runs ---
src = re.sub(r"\n{3,}", "\n\n", src)

Path("body.tex").write_text(src, encoding="utf-8")
print(f"Wrote body.tex ({len(src)} bytes, {src.count(chr(10))} lines)")
