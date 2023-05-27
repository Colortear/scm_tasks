NAME := cmp_vec
CC := nvcc
OFLAGS := -lcublas
FLAGS :=
SRC = cmp_vec.cu
OBJ = $(SRC:.cu=.o)

%.o: %.cu
	$(CC) $(OFLAGS) -c $< -o $@

$(NAME): $(OBJ)
	$(CC) $(FLAGS) -o $@ $^

all: $(NAME)

debug: all
	$(DEBUG)

clean:
	$(RM) -r $(OBJ)

fclean: clean
	$(RM) $(NAME)

.PHONY: all fclean clean re
