CC ?= gcc
CFLAGS = -Iinclude -std=c99
CLIB = -lcfitsio -lOpenCL
# CLIB = -framework OpenCL -lcfitsio

# reduce: src/reduce.c
# 	${CC} ${CFLAGS} src/reduce.c -o bin/reduce ${CLIB}

.c.o:
	$(CC) $(CFLAGS) -o $*.o -c $*.c

turnstile: src/turnstile.c src/kepler.o
	${CC} ${CFLAGS} src/turnstile.c src/kepler.o -o bin/turnstile ${CLIB}
