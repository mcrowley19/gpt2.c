test:
	cc -Wall -Wextra -fsanitize=address tests/test_arena.c -o test_arena
	./test_arena