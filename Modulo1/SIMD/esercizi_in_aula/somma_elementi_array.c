#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>

#define VECTOR_LENGTH 32 //1048576 // 1024 * 1024 
#define CHAR_DATA_LANES 16

int main() {
    srand(time(NULL));

    unsigned char* A  = (unsigned char*)_mm_malloc (VECTOR_LENGTH, sizeof(unsigned char));  // 16 byte (128 bit) = taglia del registro esteso aligned

    for (int i=0; i < VECTOR_LENGTH; i++) {   
        A[i] = rand() % 256;
    }

    /* --- VERSIONE SEQUENZIALE ---*/
    unsigned int sum = 0;

    uint64_t clock_counter_start = __rdtsc();
    for (int i=0; i < VECTOR_LENGTH; i++) {   
        sum += A[i];
    } 
    uint64_t clock_counter_end = __rdtsc();

    printf("\nSequential output data:\n");
    printf("%d\n", sum);
    printf("Elapsed clocks: %lu\n", clock_counter_end-clock_counter_start);

    /* --- VERSIONE SIMD ---*/
    // puntatori alla memoria centrale, __m128i perchè devo prendere 16 byte alla volta
    __m128i *p_A = (__m128i*) A;

    // registri estesi utilizzati
    __m128i batch_8_bit;
    __m128i batch_16_bit_high;
    __m128i batch_16_bit_low;
    __m128i addendi;
    __m128i accumuli = _mm_set_epi32(0, 0, 0, 0);
    // mi serve per estendere a 16 bit le lane mediante unpack   
    __m128i zero_register = _mm_set_epi32(0, 0, 0, 0);

    clock_counter_start = __rdtsc();
    for(int i=0; i < VECTOR_LENGTH/CHAR_DATA_LANES; i++) {
        batch_8_bit = _mm_load_si128(p_A + i); 
        
        batch_16_bit_high = _mm_unpackhi_epi8(batch_8_bit, zero_register);
        batch_16_bit_low = _mm_unpacklo_epi8(batch_8_bit, zero_register);
        // sommo considerando le lane sopra che in teoria sono da 8 bit come se fossero da 16
        addendi = _mm_hadd_epi16(batch_16_bit_high, batch_16_bit_low);  
        accumuli = _mm_add_epi16(, __m128i b)
    }
    clock_counter_end = __rdtsc();

    printf("\nSIMD output data:\n");
    print_output(C);
    printf("\nElapsed clocks: %lu\n", clock_counter_end-clock_counter_start);

    _mm_free(A);
}