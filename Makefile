METHOD ?= m1
TARGET = sw

CX = g++
CXXFLAGS = -O3 -Wall -fopenmp -mavx2 -std=c++11


ifeq ($(METHOD), m1)
    SRCS = sw_rowmajor.cc
else ifeq ($(METHOD), m2)
    SRCS = sw_alldiag.cc
else ifeq ($(METHOD), m3)
    SRCS = sw_2diag.cc
else ifeq ($(METHOD), m4)
    SRCS = sw_table.cc
else ifeq ($(METHOD), m5)
    SRCS = sw_simd.cc
else ifeq ($(METHOD), m6)
    SRCS = origin.cc
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

