#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>

#define VECTOR_LENGTH 48 
#define CHAR_DATA_LANES 16

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

  char A[VECTOR_LENGTH] __attribute__((aligned(CHAR_DATA_LANES)));  // 16 byte (128 bit) = taglia del registro esteso aligned
  char B[VECTOR_LENGTH] __attribute__((aligned(CHAR_DATA_LANES)));  // 16 byte (128 bit) = taglia del registro esteso aligned
  char C[VECTOR_LENGTH] __attribute__((aligned(CHAR_DATA_LANES)));  // 16 byte (128 bit) = taglia del registro esteso aligned

  for (int i=0; i < VECTOR_LENGTH; i++) {   
      A[i] = rand()%256 - 128;
      B[i] = rand()%256 - 128;
  }

  printf("\nInput data:\n");
  print_input(A, B);

  /* --- VERSIONE SEQUENZIALE ---*/
  uint64_t clock_counter_start = __rdtsc();
  for (int i=0; i < VECTOR_LENGTH; i++) {   
      if(A[i] < B[i]) 
        C[i] = A[i];
      else 
        C[i] = B[i]; 
  } 
  uint64_t clock_counter_end = __rdtsc();

  printf("\nSequential output data:\n");
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
  for(int i=0; i < VECTOR_LENGTH/CHAR_DATA_LANES; i++) {
      e_register_A = _mm_load_si128(p_A + i); 
      e_register_B = _mm_load_si128(p_B + i);
      // maschera dei minori (OxFF se la lane in A è minore della lane in B)
      e_register_mask = _mm_cmplt_epi8(e_register_A, e_register_B);
      //registro con i minimi blended trovati applicando la maschera ai due registri (faccio passare il primo se la maschera è a 0 se no il secondo)
      e_register_C = _mm_blendv_epi8(e_register_B, e_register_A, e_register_mask);
      _mm_store_si128(p_C + i, e_register_C);      
  }
  clock_counter_end = __rdtsc();

  printf("\nSIMD output data:\n");
  print_output(C);
  printf("\nElapsed clocks: %lu\n", clock_counter_end-clock_counter_start);

  /* --- VERSIONE SIMD DEL PROF ---*/
  clock_counter_start = __rdtsc();
  for(int i=0; i < VECTOR_LENGTH/CHAR_DATA_LANES; i++) {
      e_register_A = _mm_load_si128(p_A + i); 
      e_register_B = _mm_load_si128(p_B + i);
      // maschera dei minori (OxFF se la lane in A è minore della lane in B)
      e_register_mask = _mm_cmplt_epi8(e_register_A, e_register_B);
      // contiene i soli valori di A che risultano minori (nella linea giusta)
      __m128i tmp1 = _mm_and_si128(e_register_A, e_register_mask);
      // contiene i soli valori di B che risultano minori (nella linea giusta)
      __m128i tmp2 = _mm_andnot_si128(e_register_mask, e_register_B);
      // unisco i due con un or
      e_register_C = _mm_or_si128(tmp1, tmp2);

      _mm_store_si128(p_C + i, e_register_C);      
  }
  clock_counter_end = __rdtsc();

  printf("\nSIMD PROF output data:\n");
  print_output(C);
  printf("\nElapsed clocks: %lu\n", clock_counter_end-clock_counter_start);

  /*
    NB: nella versione del prof, dopo la compare, uso 3 istruzioni a cui corrispondono 3 clock al posto dei 2 clock che impiega la blendv 
    nella mia soluzione. Tuttavia la soluzione del prof è più generale, in quanto blendv non è disponibili con interi a 16 bit per esempio 
    (sarebbe stato possibile usare blend con l'immediato?). 
   */
}