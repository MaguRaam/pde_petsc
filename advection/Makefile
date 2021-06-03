CFLAGS     = -Wall
TARGET     = ex1
SRC        = $(TARGET).c
OBJ        = $(SRC:.c=.o)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: $(TARGET)

$(TARGET): $(OBJ)
	-${CLINKER} -o $(TARGET) $(OBJ) ${PETSC_LIB}
	${RM} $(OBJ)
