# include <assert.h>
# include "../src/arena.c"

int main(void){
    Arena a;
    arena_init(&a, 1024);

    unsigned char *p1 = arena_alloc(&a, 100);
    unsigned char *p2 = arena_alloc(&a, 100);

    assert(p2 == p1 + 100);
}