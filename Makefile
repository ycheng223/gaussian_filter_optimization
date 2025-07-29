# gcc -O2 -Wall -Wextra -I inc/lodepng src/main.c inc/lodepng/lodepng.c -o gaussian_filter -lm


TARGET = gaussian_filter
SRC = src/main.c inc/lodepng/lodepng.c
CC = gcc
CFLAGS = -O2 -Wall -Wextra -Iinc/lodepng -mssse3 -msse4.1
LDFLAGS = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
