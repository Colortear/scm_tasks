SUB := $(wildcard */*/.)

all: $(SUB)

$(SUB):
	$(MAKE) -C $@

.PHONY: all clean fclean $(SUB)

clean:
	@for dir in $(SUB); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done

fclean:
	@for dir in $(SUB); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done
