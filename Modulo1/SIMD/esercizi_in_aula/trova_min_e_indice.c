#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <limits.h>

#define VECTOR_LENGTH 1048576 // 1024 * 1024 
#define CHAR_DATA_LANES 16

void print_output(char *A) {
  for (int i=0; i<VECTOR_LENGTH; i++) {   
      printf("A[%d]=%d\n", i, A[i]);
  }
}

int main() {
  srand(time(NULL));

  char* A  = (char*)_mm_malloc (VECTOR_LENGTH, sizeof(char));  // 16 byte (128 bit) = taglia del registro esteso aligned

  for (int i=0; i < VECTOR_LENGTH; i++) {   
      A[i] = rand()%256 - 128;
  }

//   printf("\nInput data:\n");
//   print_output(A);

  /* --- VERSIONE SEQUENZIALE ---*/
  char scalar_prev_min = 127;
  int indice_min = 0;

  uint64_t clock_counter_start = __rdtsc();
  for (int i=0; i < VECTOR_LENGTH; i++) {   
      if(A[i] < scalar_prev_min) {
        scalar_prev_min = A[i];
        indice_min = i;
      }
  } 
  uint64_t clock_counter_end = __rdtsc();
  uint64_t elapsed_seq = clock_counter_end-clock_counter_start;

  printf("\nSequential:\t (min: %d; indice: %d)\n", scalar_prev_min, indice_min);
  printf("Elapsed clocks: %lu\n", elapsed_seq);

  /* --- VERSIONE SIMD ---*/
  // puntatore alla memoria centrale, __m128i perchè devo prendere 16 byte alla volta
  __m128i *p_A = (__m128i*) A;

  // registri estesi utilizzati
  __m128i prev_min = _mm_set_epi32 (INT_MAX, INT_MAX, INT_MAX, INT_MAX);
  __m128i new_data;
  __m128i mask;

  clock_counter_start = __rdtsc();
  for(int i=0; i < VECTOR_LENGTH/CHAR_DATA_LANES; i++) {
      new_data = _mm_load_si128(p_A + i); 
      // maschera dei minori (OxFF se la lane in new_data è minore della lane in prev_min)
      mask = _mm_cmplt_epi8(new_data, prev_min);
      // contiene i soli valori di new_data che risultano minori (nella linea giusta)
      __m128i tmp1 = _mm_and_si128(new_data, mask);
      // contiene i soli valori di prev_min che risultano minori (nella linea giusta)
      __m128i tmp2 = _mm_andnot_si128(mask, prev_min);
      // unisco i due con un or
      prev_min = _mm_or_si128(tmp1, tmp2);
  }
  
  // reduce del contenuto del registro finale 
  char results[CHAR_DATA_LANES];
  _mm_store_si128((__m128i*)results, prev_min);
  char result = 127;
  for(int i=0; i<CHAR_DATA_LANES; i++) {
     if(results[i] < result) 
        result = results[i];
  }
  clock_counter_end = __rdtsc();
  uint64_t elapsed_SIMD = clock_counter_end-clock_counter_start;

  printf("\nSIMD min: %d\n", result);
  printf("Elapsed clocks: %lu\n", elapsed_SIMD);
  printf("\nSPEEDUP: %f\n", (float)elapsed_seq/elapsed_SIMD);

  _mm_free(A);
}