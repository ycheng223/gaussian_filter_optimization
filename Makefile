# g++ -std=c++11 -O2 -Wall -Wextra -I inc/lodepng src/main.c inc/lodepng/lodepng.cpp -o gaussian_filter -lm


TARGET = gaussian_filter
SRC = src/main.c inc/lodepng/lodepng.cpp
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -Iinc/lodepng
LDFLAGS = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean 