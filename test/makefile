
CC		=	g++
CXXFLAGS	=	-std=c++14 -Wall -O3
TARGET		=	thomas

all : $(TARGET)

$(TARGET) : $(TARGET).cc
	$(CC) $(CXXFALGS) -o $(TARGET) $(TARGET).cc

.PHONY: clean

clean:
	rm -f $(TARGET) *.vtk *.dat *.png