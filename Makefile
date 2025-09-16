# gcc -O2 -Wall -Wextra -mssse3 -msse4.1 -I inc/lodepng src/main.c inc/lodepng/lodepng.c -o gaussian_filter -lm


TARGET = gaussian_filter

C_SRC = src/main.c \
      src/gaussian_filter.c \
      src/gaussian_processing.c \
      src/utility.c \
      src/image_operations.c \
      src/png_transform.c \
      src/debug.c \
      inc/lodepng/lodepng.c

CU_SRC = src/gaussian_filter_cuda.cu

CC = gcc
NVCC = nvcc

CFLAGS = -O2 -Wall -Wextra -Iinc -Iinc/lodepng -mssse3 -msse4.1 -I/usr/local/cuda/include
NVCCFLAGS = -O2 -Iinc -I/usr/local/cuda/include
LDFLAGS = -lm -lcuda -lcudart -L/usr/local/cuda/lib64

# Object files
C_OBJS = $(C_SRC:.c=.o)
CU_OBJS = $(CU_SRC:.cu=.o)

all: $(TARGET)

$(TARGET): $(C_OBJS) $(CU_OBJS)
	$(NVCC) $(C_OBJS) $(CU_OBJS) -o $(TARGET) $(LDFLAGS)

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA files  
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJS) $(CU_OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
