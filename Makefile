METHOD ?= sw_rowmajor
TARGET = sw

CX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp -mavx2 -std=c++11


ifeq ($(METHOD), sw_rowmajor)
    SRCS = sw_rowmajor.cc
else ifeq ($(METHOD), sw_alldiag)
    SRCS = sw_alldiag.cc
else ifeq ($(METHOD), sw_2diag)
    SRCS = sw_2diag.cc
else ifeq ($(METHOD), sw_row_simd)
    SRCS = sw_row_simd.cc
else ifeq ($(METHOD), sw_simd)
    SRCS = sw_simd.cc
else ifeq ($(METHOD), sw_origin)
    SRCS = origin.cc
else ifeq ($(METHOD), sw_simd2)
    SRCS = sw_simd2.cc
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
