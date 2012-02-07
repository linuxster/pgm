include Makefile.inc

NAME    = ps1
DOC_OUT = ${NAME}.${SUFF}

DIRS    = gm

default: subdirs ${DOC_OUT}

%.pdf: %.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || ${ECHO} "Done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || ${ECHO} "Done."

subdirs: force_look
	@$(foreach d, ${DIRS}, (echo "Looking into ${d}:"; cd ${d}; ${MAKE} ${MFLAGS}) );

clean:
	${RM_TMP}
	@$(foreach d, ${DIRS}, (echo "Cleaning ${d}:"; cd ${d}; $(MAKE) clean) );

force_look:
	true

