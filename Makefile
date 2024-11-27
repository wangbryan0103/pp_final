TARGET = sw

CX = g++
CXXFLAGS = -O3 -Wall -fopenmp -mavx2 -std=c++11

SRC = sw_simd.cc
OBJ = $(SRC:.cc=.o)

$(TARGET): $(OBJ)
	$(CX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cc
	$(CX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)
	rm -f $(OBJ)
	rm -f sw_wavefront.o
	rm -f sw_2diag.o
