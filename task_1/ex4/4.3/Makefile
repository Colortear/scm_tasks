NAME := tri_diag
CC := nvcc
OFLAGS := -lcublas
FLAGS := -lcublas
SRC = main.cu cuBLAS_Vec.cu cuBLAS_Op.cu
OBJ = $(SRC:.cu=.o)
DEBUG := -g

%.o: %.cu
	$(CC) $(OFLAGS) -I. -c $< -o $@

$(NAME): $(OBJ)
	$(CC) $(FLAGS) -I. -o $@ $^

all: $(NAME)

debug: OFLAGS += -g
debug: FLAGS += -g
debug: $(NAME)

clean:
	$(RM) -r $(OBJ)

fclean: clean
	$(RM) $(NAME)

.PHONY: all fclean clean re
