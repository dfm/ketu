CC ?= gcc
CLIB = -framework OpenCL

reduce: src/reduce.c
	${CC} ${CFLAGS} src/reduce.c -o bin/reduce ${CLIB}
