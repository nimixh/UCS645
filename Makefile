CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp

TARGET = correlate

SOURCES = main.cpp correlate.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

run:
	./$(TARGET) 1000 1000

.PHONY: all clean run
