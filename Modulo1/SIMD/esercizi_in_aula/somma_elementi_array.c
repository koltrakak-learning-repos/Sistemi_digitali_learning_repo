#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>

#define VECTOR_LENGTH  4194304  // 2048 * 2048 
#define CHAR_DATA_LANES 16
#define SHORT_DATA_LANES 8
#define INT_DATA_LANES 4

void print_array(uint8_t *A, int dim) {
  for (int i=0; i<dim; i++) {   
      printf("array[%d]=%d\n", i, A[i]);
  }
}

void print_array_short(uint16_t *A, int dim) {
  for (int i=0; i<dim; i++) {   
      printf("array[%d]=%d\n", i, A[i]);
  }
}

int main() {
    srand(time(NULL));

    uint8_t* A  = (uint8_t*)_mm_malloc (VECTOR_LENGTH, sizeof(CHAR_DATA_LANES));  // 16 byte (128 bit) = taglia del registro esteso aligned
    for (int i=0; i < VECTOR_LENGTH; i++) {   
        A[i] = rand() % 256;
    }
    // print_array(A, VECTOR_LENGTH);



    /* --- VERSIONE SEQUENZIALE ---*/

    

    uint32_t sum = 0;

    uint64_t clock_counter_start = __rdtsc();
    for (int i=0; i < VECTOR_LENGTH; i++) {   
        sum += A[i];
    } 
    uint64_t clock_counter_end = __rdtsc();
    uint64_t sequential_elapsed = clock_counter_end-clock_counter_start;

    printf("\nSequential output data: %d\n", sum);
    printf("Elapsed clocks: %lu\n\n", sequential_elapsed);



    /* --- VERSIONE SIMD 8-LANE ---*/



    sum = 0;
    __m128i *p_A = (__m128i*) A;
     
    // registri estesi utilizzati
    __m128i batch_8_bit;
    __m128i batch_16_bit_high;
    __m128i batch_16_bit_low;
    __m128i addendi;
    __m128i accumuli = _mm_set_epi32(0, 0, 0, 0);
    // mi serve per estendere a 16 bit le lane mediante unpack   
    __m128i zero_register = _mm_set_epi32(0, 0, 0, 0);

    // Somma vera e propria, nel caso peggiore (array di tutti 255) accumulo ogni volta 510
    // Posso fare al massimo 128 somme essendo sicuro di non andare in overflow ( [2^16-1]/510 = 128,5 )
    // HP: considero vettori di dimensione multipla di 2048 (128*16) e accumulo per blocchi di questa dimensione  
    uint16_t array_accumuli[SHORT_DATA_LANES];

    clock_counter_start = __rdtsc();
    for(int i=0; i < VECTOR_LENGTH/(CHAR_DATA_LANES*128); i++) {
        //qua dentro sto considerando l'i-esimo blocco da 128 elementi dell'array

        for(int j = i*128; j < (i*128)+128; j++) {
            batch_8_bit = _mm_load_si128(p_A + j); 
            // estendo a 16 bit senza segno in due registri facendo interleaving
            batch_16_bit_low = _mm_unpacklo_epi8(batch_8_bit, zero_register);
            batch_16_bit_high = _mm_unpackhi_epi8(batch_8_bit, zero_register);
            // sommo orizzontalmente a due a due considerando le lane sopra (che in teoria sono da 8 bit) come se fossero da 16
            // ottengo un registro con 8 lane
            addendi = _mm_hadd_epi16(batch_16_bit_low, batch_16_bit_high);  
            // accumulo i risultati
            accumuli = _mm_add_epi16(accumuli, addendi);
        }

        //salvo il risultato parziale ed azzero gli accumuli
        _mm_store_si128((__m128i*)array_accumuli, accumuli);
        for(int i=0; i<SHORT_DATA_LANES; i++) {
            sum += array_accumuli[i];
        }
        accumuli = _mm_and_si128(accumuli, zero_register);
    }
    clock_counter_end = __rdtsc();
    uint64_t SIMD_elapsed = clock_counter_end-clock_counter_start;

    printf("\nSIMD output data: %d\n", sum);
    printf("Elapsed clocks: %lu\n", SIMD_elapsed);
    printf("Speedup: %0.2f\n", (double)sequential_elapsed/SIMD_elapsed);



    /* --- VERSIONE SIMD 4-LANE ---*/



    __m128i batch_32_bit_low_low;
    __m128i batch_32_bit_low_high;
    __m128i batch_32_bit_high_low;
    __m128i batch_32_bit_high_high;

    uint32_t array_accumuli_int[INT_DATA_LANES];
    sum = 0;
    accumuli = _mm_set_epi32(0, 0, 0, 0);

    clock_counter_start = __rdtsc();
    for(int i=0; i < VECTOR_LENGTH/CHAR_DATA_LANES; i++) {
        // considero 16 dati alla volta
        batch_8_bit = _mm_load_si128(p_A + i); 

        // estendo a 32 bit facendo interleaving 2 volte
            batch_16_bit_low = _mm_unpacklo_epi8(batch_8_bit, zero_register);
        batch_32_bit_low_low = _mm_unpacklo_epi16(batch_16_bit_low, zero_register);
        batch_32_bit_low_high = _mm_unpackhi_epi16(batch_16_bit_low, zero_register);

            batch_16_bit_high = _mm_unpackhi_epi8(batch_8_bit, zero_register);
        batch_32_bit_high_low = _mm_unpacklo_epi16(batch_16_bit_high, zero_register);
        batch_32_bit_high_high = _mm_unpackhi_epi16(batch_16_bit_high, zero_register);

        // sommo orizzontalmente a due a due considerando le lane sopra (che in teoria sono da 8 bit) come se fossero da 32
        __m128i addendi_low = _mm_hadd_epi32(batch_32_bit_low_low, batch_32_bit_low_high);  
        __m128i addendi_high = _mm_hadd_epi32(batch_32_bit_high_low, batch_32_bit_high_high);  
        addendi = _mm_add_epi32(addendi_low, addendi_high);
        // accumulo i risultati
        accumuli = _mm_add_epi32(accumuli, addendi);
    }
    // faccio l'ultima somma
    _mm_store_si128((__m128i*)array_accumuli_int, accumuli);
    for(int i=0; i<INT_DATA_LANES; i++) {
        sum += array_accumuli_int[i];
    }

    clock_counter_end = __rdtsc();
    uint64_t SIMD_elapsed_int = clock_counter_end-clock_counter_start;

    printf("\nSIMD a 4-lane output data: %d\n", sum);
    printf("Elapsed clocks: %lu\n", SIMD_elapsed_int);
    printf("Speedup: %0.2f\n", (double)sequential_elapsed/SIMD_elapsed_int);

    _mm_free(A);
}