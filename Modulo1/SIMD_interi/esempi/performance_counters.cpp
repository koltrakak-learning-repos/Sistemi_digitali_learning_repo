#include <stdio.h>
#include <immintrin.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define VECTOR_LENGTH 1024 
#define SSE_DATA_LANE 16
#define DATA_SIZE 1

void print_output(char *A, char *B, int length)
{
  for (int i=0; i<VECTOR_LENGTH; i++)
  {   
      printf("A[%d]=%d, B[%d]=%d\n",i,A[i],i,B[i]);
  }
}

int main()
{
  
u_int64_t clock_counter_scalar_start, clock_counter_scalar_end;
u_int64_t clock_counter_SIMD_start, clock_counter_SIMD_end;

clock_counter_scalar_start = __rdtsc();  
  
  char A[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));  // 16 byte = 128 bit aligned
  char B[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));

  __m128i *p_A = (__m128i*) A;
  __m128i *p_B = (__m128i*) B;

  __m128i XMM_SSE_REG;

for (int i=0; i<VECTOR_LENGTH; i++)
{   
    A[i]=i;
    B[i]=0;
}

clock_counter_scalar_start = __rdtsc(); 

for (int i=0; i<VECTOR_LENGTH; i++)
{   
    B[i]=A[i];
}

clock_counter_scalar_end = __rdtsc(); 

printf("\nInput data:\n");
print_output(A,B,VECTOR_LENGTH);

clock_counter_SIMD_start = __rdtsc(); 

for (int i=0; i<VECTOR_LENGTH/SSE_DATA_LANE/DATA_SIZE; i++)
{
  XMM_SSE_REG = _mm_load_si128 (p_A+i);
  _mm_store_si128 (p_B+i, XMM_SSE_REG);      
}

clock_counter_SIMD_end = __rdtsc(); 

printf("\nOutput data:\n");
print_output(A,B,VECTOR_LENGTH);

printf("\nElapsed clocks (scalar): %lu\n", clock_counter_scalar_end-clock_counter_scalar_start);
printf("Elapsed clocks (SIMD): %lu\n", clock_counter_SIMD_end-clock_counter_SIMD_start);
printf("Speed-up = %3.2f\n", (clock_counter_scalar_end - clock_counter_scalar_start)/((clock_counter_SIMD_end - clock_counter_SIMD_start)*1.0));

}