.PHONY : tests

all :
	echo "Configure your own targets here."

tests :
	py.test -v --cov util --cov-report html
