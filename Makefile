METHOD ?= sw_row_pthread
TARGET = sw

CX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp -mavx2 -std=c++11


ifeq ($(METHOD), sw_row_pthread)
    SRCS = sw_row_pthread.cc
else ifeq ($(METHOD), sw_diag_openmp)
    SRCS = sw_diag_openmp.cc
else ifeq ($(METHOD), sw_row_simd)
    SRCS = sw_row_simd.cc
else ifeq ($(METHOD), sw_diag_simd)
    SRCS = sw_diag_simd.cc
endif

OBJ = $(SRCS:.cc=.o)

$(TARGET): $(OBJ)
	$(CX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cc
	$(CX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR)
	rm -f *.o
