
#include "arena.c"
#define VOCAB_SIZE 50257
typedef struct{
    int a;
    int b;
    int rank;
    int id;
} HashmapEntry;

typedef struct{
    int table_size;
    HashmapEntry entries[];
} Hashmap;


int hash(Hashmap *h, int a, int b){
    return ((long) a * VOCAB_SIZE + b) % h->table_size;
}

void insert(Hashmap *h, int a, int b, int rank, int id){
    int index = hash(h, a, b);
    h->entries[index].a  = a;
    h->entries[index].b  = b;
    h->entries[index].rank = rank;
    h->entries[index].id = id;
}

Hashmap* init_hashmap(int table_size, Arena *arena){
    Hashmap *res = arena_alloc(arena, sizeof(Hashmap) + sizeof(HashmapEntry) * table_size);
    for(int i = 0; i < table_size; i++){
        res->entries[i].a = -1;
    }
    res->table_size = table_size;
    return res;
}

HashmapEntry* lookup(Hashmap *h, int a, int b){
    int index = hash(h, a, b);
    HashmapEntry *entry = &(h->entries[index]);
    if(entry->a == a && entry->b == b){
        return entry;
    }
    return NULL;
}