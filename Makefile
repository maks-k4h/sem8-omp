CXX = clang++
CXXFLAGS = -std=c++11 -Wall -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include $(shell pkg-config --cflags opencv4)
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp $(shell pkg-config --libs opencv4)
TARGET = kmeans_omp
INPUT = input.png
OUTPUT = output_omp.png

.PHONY: all build run clean

all: build

build:
	$(CXX) $(CXXFLAGS) -o $(TARGET) kmeans_omp.cpp $(LDFLAGS)

run:
	@if [ -f $(INPUT) ]; then \
		OMP_NUM_THREADS=4 ./$(TARGET); \
	else \
		echo "Error: Input file $(INPUT) not found"; exit 1; \
	fi

clean:
	rm -f $(TARGET) $(OUTPUT)

test: build
	@if [ -f $(INPUT) ]; then \
		echo "Running with 4 threads..."; \
		OMP_NUM_THREADS=4 ./$(TARGET) && echo "Success" || echo "Failed"; \
		[ -f $(OUTPUT) ] && echo "Output created" || echo "No output"; \
	else \
		echo "Error: Missing input file"; exit 1; \
	fi