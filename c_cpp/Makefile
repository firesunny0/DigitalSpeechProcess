.PHONY: all clean

CFLAGS+= -std+=C++11
LDFLAGS+=-lm     # link to math library

SRC+=./*.c ./*.cpp
TARGET=test_hmm

all: $(TARGET)
	g++ SRC -I ./ -o test_hmm


# type make/make all to compile test_hmm

clean:
	$(RM) $(TARGET)   # type make clean to remove the compiled file
