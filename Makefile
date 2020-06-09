.PHONY: test clean

PLY_COMPILED := $(shell find rl_parsers/ -type f \( -name lextab.py -o -name parsetab.py -o -name parser.out \))

test:
	python -m unittest discover

clean:
	rm -f ${PLY_COMPILED}
