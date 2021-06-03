include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


SRC				=		$(wildcard *.c)
OBJ 			= 	$(SRC:.c=.o)
TARGET		=		main		

$(TARGET): $(OBJ)
	-${CLINKER} $^ -o $@ $(PETSC_LIB)

$(OBJ): $(SRC)
	-${CLINKER} -c $< $(PETSC_CC_INCLUDES)

print_vars:
	-@echo "PETSC_DIR: $(PETSC_DIR)"
	-@echo "CLINKER: $(CLINKER)"
	-@echo "CXX_COMPILE: $(PETSC_CXXCOMPILE)"
	-@echo "C_COMPILE: $(PETSC_COMPILE)"

distclean :
	@$(RM) $(TARGET) $(OBJ)
