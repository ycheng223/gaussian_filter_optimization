# gcc -O2 -Wall -Wextra -mssse3 -msse4.1 -I inc/lodepng src/main.c inc/lodepng/lodepng.c -o gaussian_filter -lm


TARGET = gaussian_filter

SRC = src/main.c \
      src/gaussian_filter.c \
      src/gaussian_processing.c \
      src/utility.c \
      src/image_operations.c \
      src/png_transform.c \
      src/debug.c \
      inc/lodepng/lodepng.c

CC = gcc
CFLAGS = -O2 -Wall -Wextra -Iinc -Iinc/lodepng -mssse3 -msse4.1
LDFLAGS = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
