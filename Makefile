
archivo=third
biblio=third
PUBLISH=slides/

# all: compile bibliography recompile open
all: compile

clean:
	rm -rf $(PUBLISH)*.log $(PUBLISH)*.toc $(PUBLISH)*.toc $(PUBLISH)*.out
	rm -rf $(PUBLISH)*.nav $(PUBLISH)*.aux $(PUBLISH)*.snm
	rm -rf $(PUBLISH)*.vrb

compile:
	lualatex -output-directory $(PUBLISH) ep2018.tex

#bibliography:
#	bibtex publish/gutierrez_cv

read:
	evince $(PUBLISH)ep2018.pdf &
