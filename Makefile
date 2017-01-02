.PHONY : tests

all :
	echo "Configure your own targets here."

tests :
	MPLBACKEND=tkagg py.test -v --cov util --cov-report html
