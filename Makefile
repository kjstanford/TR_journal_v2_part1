MAIN := main
LOGDIR := logs
TEX := pdflatex
BIBTEX := bibtex
FLAGS := -interaction=nonstopmode -halt-on-error -output-directory=$(LOGDIR)

.PHONY: all clean distclean

all: $(MAIN).pdf

$(LOGDIR):
	mkdir -p $(LOGDIR)

$(MAIN).pdf: $(MAIN).tex | $(LOGDIR)
	$(TEX) $(FLAGS) $(MAIN).tex
	@if grep -q '\\bibliography' $(MAIN).tex && [ -f $(LOGDIR)/$(MAIN).aux ]; then \
		$(BIBTEX) $(LOGDIR)/$(MAIN) || true; \
		$(TEX) $(FLAGS) $(MAIN).tex; \
	fi
	$(TEX) $(FLAGS) $(MAIN).tex
	cp $(LOGDIR)/$(MAIN).pdf $(MAIN).pdf

clean:
	rm -rf $(LOGDIR)

distclean: clean
	rm -f $(MAIN).pdf
