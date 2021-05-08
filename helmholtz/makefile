include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules



TARGET = ex2

all: $(TARGET)

$(TARGET) : $(TARGET).cc
	-${CLINKER}  $(TARGET).cc -o $(TARGET) ${PETSC_LIB}
	$(RM) $(TARGET).o

distclean :
	$(RM) $(TARGET)
