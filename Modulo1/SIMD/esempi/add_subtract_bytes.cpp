#include <stdio.h>
#include <immintrin.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// #define VERBOSE OUTPUT


#define VECTOR_LENGTH 16 
#define SSE_DATA_LANE 16
#define DATA_SIZE 1

int main()
{
 
  printf("Load and store with SIMD SSE\n\n");

  char A[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));  // 16 byte = 128 bit aligned
  char B[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
  char C[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));  

  unsigned char UA[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));  // 16 byte = 128 bit aligned
  unsigned char UB[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));
  unsigned char UC[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));  

  int i;

 
  __m128i *p_UA = (__m128i*) UA;
  __m128i *p_UB = (__m128i*) UB;
  __m128i *p_UC = (__m128i*) UC;  

  __m128i *p_A = (__m128i*) A;
  __m128i *p_B = (__m128i*) B;
  __m128i *p_C = (__m128i*) C;  

  __m128i XMM_SSE_REG_A;
  __m128i XMM_SSE_REG_B;
  __m128i XMM_SSE_REG_C;

  printf("\n *** Unsigned byte ***\n");

  XMM_SSE_REG_A = _mm_set_epi8(200,64,250,4,45,128,77,6,87,7,68,195,0,255,128,3);
  XMM_SSE_REG_B = _mm_set_epi8(100,68,210,34,145,120,68,26,87,71,32,106,30,53,128,103);

  _mm_storeu_si128 (p_UA, XMM_SSE_REG_A);
  _mm_storeu_si128 (p_UB, XMM_SSE_REG_B);

  printf("\nUA = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%hhu ",UA[i]);
  }

 printf("\nUB = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%hhu ",UB[i]);
  }

  XMM_SSE_REG_C = _mm_add_epi8 (XMM_SSE_REG_A, XMM_SSE_REG_B);
  _mm_storeu_si128 (p_UC, XMM_SSE_REG_C);

 printf("\n\nADD:");
 printf("\nUC = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%hhu ",UC[i]);
  }

  XMM_SSE_REG_C = _mm_sub_epi8 (XMM_SSE_REG_A, XMM_SSE_REG_B);
  _mm_storeu_si128 (p_UC, XMM_SSE_REG_C);

 printf("\n\nSUBTRACT:");
 printf("\nUC = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%hhu ",UC[i]);
  }


 printf("\n\n *** Signed byte ***\n");

  XMM_SSE_REG_A = _mm_set_epi8(-20,64,50,4,-45,127,-77,6,-87,7,68,-95,0,15,-127,3);
  XMM_SSE_REG_B = _mm_set_epi8(100,68,10,34,-127,-128,68,26,87,71,32,106,-30,53,-127,-103);

  _mm_storeu_si128 (p_A, XMM_SSE_REG_A);
  _mm_storeu_si128 (p_B, XMM_SSE_REG_B);

  printf("\nA = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%d ",A[i]);
  }

 printf("\nB = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%d ",B[i]);
  }

  XMM_SSE_REG_C = _mm_add_epi8 (XMM_SSE_REG_A, XMM_SSE_REG_B);
  _mm_storeu_si128 (p_C, XMM_SSE_REG_C);

 printf("\n\nADD:");
 printf("\nC = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%d ",C[i]);
  }

  XMM_SSE_REG_C = _mm_sub_epi8 (XMM_SSE_REG_A, XMM_SSE_REG_B);
  _mm_storeu_si128 (p_C, XMM_SSE_REG_C);

 printf("\n\nSUBTRACT:");
 printf("\nC = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%d ",C[i]);
  }

  XMM_SSE_REG_C = _mm_adds_epi8 (XMM_SSE_REG_A, XMM_SSE_REG_B);
  _mm_storeu_si128 (p_C, XMM_SSE_REG_C);

 printf("\n\nADD (Saturation):");
 printf("\nC = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%d ",C[i]);
  }

  XMM_SSE_REG_C = _mm_subs_epi8 (XMM_SSE_REG_A, XMM_SSE_REG_B);
  _mm_storeu_si128 (p_C, XMM_SSE_REG_C);

 printf("\n\nSUBTRACT (Saturation):");
 printf("\nC = ");

  for (i=VECTOR_LENGTH-1; i>=0; i--)
  {
    printf("\t%d ",C[i]);
  }



}