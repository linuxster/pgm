LATEX       = pdflatex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
RM_TMP      = ${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})

TMP_SUFFS   = pdf aux bbl blg log dvi ps eps
SUFF        = pdf

CHECK_RERUN = grep Rerun $*.log

NAME    = ps3
DOC_OUT = ${NAME}.${SUFF}

default: ${DOC_OUT}

%.pdf: %.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || ${ECHO} "Done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || ${ECHO} "Done."

clean:
	${RM_TMP}

