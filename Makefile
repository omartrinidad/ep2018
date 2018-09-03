
archivo=linear_algebra_ml
biblio=linear_algebra_ml
PUBLISH=slides/

# all: compile bibliography recompile open
all: compile

clean:
	rm -rf $(PUBLISH)*.log $(PUBLISH)*.toc $(PUBLISH)*.toc $(PUBLISH)*.out
	rm -rf $(PUBLISH)*.nav $(PUBLISH)*.aux $(PUBLISH)*.snm
	rm -rf $(PUBLISH)*.vrb

compile:
	lualatex -output-directory $(PUBLISH) $(archivo).tex

#bibliography:
#	bibtex publish/gutierrez_cv

read:
	evince $(PUBLISH)$(archivo).pdf &
