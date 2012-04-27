include Makefile.inc

NAME    = ps6
DOC_OUT = ${NAME}.${SUFF}

default: ${DOC_OUT}

%.pdf: %.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "All done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "All done."

clean:
	${RM_TMP}

