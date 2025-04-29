CXX = clang++
CXXFLAGS = -std=c++11 -Xpreprocessor -fopenmp \
           -I/opt/homebrew/opt/libomp/include \
           -I/opt/homebrew/opt/opencv@4/include/opencv4
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp \
          -L/opt/homebrew/opt/opencv@4/lib \
          -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

TARGET = kmeans_omp
INPUT = input.png
OUTPUT = output.png

.PHONY: all build run clean

all: build

build:
	$(CXX) $(CXXFLAGS) -o $(TARGET) kmeans_omp.cpp $(LDFLAGS)

run:
	@if [ -f $(INPUT) ]; then \
		./$(TARGET); \
	else \
		echo "Error: Input file $(INPUT) not found"; exit 1; \
	fi

clean:
	rm -f $(TARGET) $(OUTPUT)