#include <stdio.h>
#include <immintrin.h>

#define VECTOR_LENGTH 32 
#define SSE_DATA_LANE 16
#define DATA_SIZE 1

void print_output(char *A, char *B, int length)
{
  for (int i=0; i<VECTOR_LENGTH; i++)
  {   
      printf("A[%d]=%d, B[%d]=%d\n",i,A[i],i,B[i]);
  }
}

int main() {
  char A[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));  // 16 byte (128 bit) aligned
  char B[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));  // 16 byte (128 bit) aligned

  __m128i *p_A = (__m128i*) A;
  __m128i *p_B = (__m128i*) B;

  __m128i XMM_SSE_REG;

  for (int i=0; i<VECTOR_LENGTH; i++)
  {   
      A[i]=i;
      B[i]=0;
  }

  printf("\nInput data:\n");
  print_output(A,B,VECTOR_LENGTH);

  for (int i=0; i<VECTOR_LENGTH/SSE_DATA_LANE/DATA_SIZE; i++)
  {
    XMM_SSE_REG = _mm_load_si128 (p_A+i);
    _mm_store_si128 (p_B+i, XMM_SSE_REG);      
  }

  printf("\nOutput data:\n");
  print_output(A,B,VECTOR_LENGTH);
}