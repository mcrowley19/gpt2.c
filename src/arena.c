# include <stdlib.h>
typedef struct {
    unsigned char *base;
    size_t size;
    size_t offset;
} Arena;

void arena_init(Arena *a, size_t size){
    a->base = malloc(size);
    a-> size = size;
    a -> offset = 0;
}

void* arena_alloc(Arena *a, size_t n){
    if ((a -> offset + n) <= a->size){
        void* return_val = a->base + a-> offset;
        a -> offset = a->offset + n;
        return return_val;
    }
    return NULL;
}

void arena_free(Arena *a){
    free(a->base);
}