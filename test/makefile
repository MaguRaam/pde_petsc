# Path where Blitz++ is installed
BZDIR = /home/magu/libraries/install/blitz

CXX = g++

#Flags for optimized executables
#CXXFLAGS = -std=c++14 -O2 -I$(BZDIR) -ftemplate-depth-30

#Flags for debugging
CXXFLAGS = -std=c++14 -ftemplate-depth-30 -g -DBZ_DEBUG -I$(BZDIR)/include

#blitz library
LIBS = -L$(BZDIR)/lib -lblitz -lm

#program name
TARGET = spectral

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(TARGET).cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET) *.vtk *.dat *.png
