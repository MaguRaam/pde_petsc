include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: expcircle

expcircle : expcircle.o
	-${CLINKER} -std=c++14 -o expcircle expcircle.o  ${PETSC_LIB}
