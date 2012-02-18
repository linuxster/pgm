include ../Makefile.inc

NAME    = gm3
DOC_OUT = ${NAME}.${SUFF}

default: ${DOC_OUT}

${NAME}.pdf: ${NAME}.tex
	latex ${NAME}.tex
	dvipdf ${NAME}.dvi

clean:
	${RM_TMP}

