include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


TARGET = ex3

all: $(TARGET)

$(TARGET) : $(TARGET).o
	-${CLINKER} -std=c++14 -o $(TARGET) $(TARGET).o  ${PETSC_LIB}
	$(RM) $(TARGET).o

distclean :
	$(RM) $(TARGET) *.vts
