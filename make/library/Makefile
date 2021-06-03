CC 		= cc
CFLAGS	= -g -Wall -fPIC #flag to generate .d dependency file
SRCDIR	= src
OBJDIR	= obj
LIBDIR	= lib
SRC		= $(wildcard $(SRCDIR)/*.c)
OBJ		= $(patsubst %, $(OBJDIR)/%,$(notdir $(SRC:.c=.o))) #dump object files to object dirctory
DEP		= $(OBJ:.o=.d) #dependency file for each source
LIB		= libutil.so


#Color codes:
RED 	= \033[1;31m
GREEN	= \033[1;32m
YELLOW 	= \033[1;33m
BLUE 	= \033[1;34m 
NC		= \033[1;0m

$(LIBDIR)/$(LIB): $(OBJ)
	@echo "$(RED)Creating shared library...$(NC)"
	$(CC) $(CFLAGS) -shared $^ -o $@


-include $(DEP) #include all dependency files in the makefile 

$(OBJDIR)/%.o : $(SRCDIR)/%.c
	@echo "$(GREEN) compiling... $(NC)"
	$(CC) $(CFLAGS) -MMD -c $< -o $@	


clean:
	@rm -f $(OBJ) $(LIBDIR)/$(LIB) $(DEP)