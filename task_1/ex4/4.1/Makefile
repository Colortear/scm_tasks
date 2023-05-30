NAME := vec_ops
CC := nvcc
OFLAGS := -lcublas
FLAGS := -lcublas
SRC = vec_ops.cu
OBJ = $(SRC:.cu=.o)
DEBUG := -g

%.o: %.cu
	$(CC) $(OFLAGS) -c $< -o $@

$(NAME): $(OBJ)
	$(CC) $(FLAGS) -o $@ $^

all: $(NAME)

debug: OFLAGS += -g
debug: FLAGS += -g
debug: $(NAME)

clean:
	$(RM) -r $(OBJ)

fclean: clean
	$(RM) $(NAME)

.PHONY: all fclean clean re
