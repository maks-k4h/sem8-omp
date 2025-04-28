CC = g++-12
CFLAGS = -std=c++20 -Wall -fopenmp -I/opt/homebrew/opt/libomp/include $(shell pkg-config --cflags opencv4)
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -fopenmp $(shell pkg-config --libs opencv4)
TARGET = kmeans_omp
INPUT = input.jpg
OUTPUT = output_omp.jpg

.PHONY: all build run clean test

all: build

build:
	$(CC) $(CFLAGS) -o $(TARGET) kmeans_omp.cpp $(LDFLAGS)

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
		echo "Running OpenMP version..."; \
		OMP_NUM_THREADS=4 ./$(TARGET) && echo "Test succeeded" || echo "Test failed"; \
		[ -f $(OUTPUT) ] && echo "Output file created: $(OUTPUT)" || echo "Output file not created"; \
	else \
		echo "Error: Cannot run test - input file $(INPUT) missing"; exit 1; \
	fi