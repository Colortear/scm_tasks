NAME := data_mv_GH_sys_clock
CC := nvcc
OFLAGS := -lcublas
FLAGS :=
SRC = data_mv_GH_sys_clock.cu
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
