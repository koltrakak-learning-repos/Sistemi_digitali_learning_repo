#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h>
#include <x86intrin.h>

#define VECTOR_LENGTH 16 
#define CHAR_DATA_LANES 16
#define DATA_SIZE 1

void print_input(char *A, char *B) {
  for (int i=0; i<VECTOR_LENGTH; i++) {   
      printf("A[%d]=%d; \tB[%d]=%d\n",i,A[i],i,B[i]);
  }
}

void print_output(char *C) {
  for (int i=0; i<VECTOR_LENGTH; i++) {   
      printf("C[%d]=%d\n", i, C[i]);
  }
}

int main() {
  srand(time(NULL));

  char A[VECTOR_LENGTH] __attribute__((aligned(CHAR_DATA_LANES)));  // 16 byte (128 bit) aligned
  char B[VECTOR_LENGTH] __attribute__((aligned(CHAR_DATA_LANES)));  // 16 byte (128 bit) aligned
  char C[VECTOR_LENGTH] __attribute__((aligned(CHAR_DATA_LANES)));  // 16 byte (128 bit) aligned

  for (int i=0; i < VECTOR_LENGTH; i++) {   
      A[i] = rand()%256 - 128;
      B[i] = rand()%256 - 128;
  }

  printf("\nInput data:\n");
  print_input(A, B);

  /* --- VERSIONE SEQUENZIALE ---*/
  u_int64_t clock_counter_start = __rdtsc();
  for (int i=0; i < VECTOR_LENGTH; i++) {   
      if(A[i] > B[i]) 
        C[i] = A[i];
      else 
        C[i] = B[i]; 
  } 
  u_int64_t clock_counter_end = __rdtsc();

  printf("\nSequential output data:");
  print_output(C);
  printf("Elapsed clocks: %lu\n", clock_counter_end-clock_counter_start);

  /* --- VERSIONE SIMD ---*/

  // puntatori alla memoria centrale, __m128i perchè devo prendere 16 byte alla volta
  __m128i *p_A = (__m128i*) A;
  __m128i *p_B = (__m128i*) B;
  __m128i *p_C = (__m128i*) C;

  // registri estesi utilizzati
  __m128i e_register_A;   
  __m128i e_register_B;
  __m128i e_register_C;
  __m128i e_register_mask;

  clock_counter_start = __rdtsc();
  for(int i=0; i < VECTOR_LENGTH / (CHAR_DATA_LANES/DATA_SIZE); i++) {
      e_register_A = _mm_load_si128(p_A + i); 
      e_register_B = _mm_load_si128(p_B + i);
      // maschera dei maggiori (OxFF se la lane in A è maggiore della lane in B)
      e_register_mask = _mm_cmpgt_epi8(e_register_A, e_register_B);
      //registro con i massimi blended trovati applicando la maschera ai due registri
      e_register_C = _mm_blendv_epi8(e_register_B, e_register_A, e_register_mask);
      _mm_store_si128(p_C + i, e_register_C);      
  }
  clock_counter_end = __rdtsc();

  printf("\nSIMD output data:\n");
  print_output(C);
  printf("\nElapsed clocks: %lu\n", clock_counter_end-clock_counter_start);
}