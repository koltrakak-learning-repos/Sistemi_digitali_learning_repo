#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "omp.h"

#define MAX  5
// #define DEBUG 1

void stampa_matrice(const int* mat, int dim) {
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            printf("%d\t", mat[i*dim + j]);
        }     

        printf("\n");   
    }

    printf("\n");
}



double parallel_gemm(const int* mat_a, const int* mat_b, int* mat_c, int dim) {
    double start = omp_get_wtime();

    #pragma omp parallel for schedule (static, 1)
    // i primi due cicli considerano ogni elemento della matrice
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            #ifdef DEBUG
            int thread_id = omp_get_thread_num();
            printf("\tThread %d sta calcolando mat_c[%d][%d]\n", thread_id, i, j);
            #endif
            // il singolo elemento della matrice risultante viene calcolato considerando un intera riga/colonna
            for(int k=0; k<dim; k++) {
                mat_c[i*dim + j] += mat_a[i*dim + k] * mat_b[k*dim + j];
            }  
        }     
    }

    double end = omp_get_wtime();

    return end-start;
}

int main(int argc, char** argv) {
    if(argc < 3) {
        printf("usage: ./gemm <dim_matrix> <num_thread>");
        exit(1);
    }

    int dim_matrix = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    printf("Numero massimo di thread eseguibili in parallelo: %d\n", omp_get_max_threads());
    omp_set_num_threads(num_threads);

    int size_matrix = dim_matrix*dim_matrix*sizeof(int);
    int* mat_a = malloc(size_matrix);
    int* mat_b = malloc(size_matrix);
    int* mat_c = malloc(size_matrix);
    memset(mat_c, 0, size_matrix);

    srand(time(NULL));
    for(int i=0; i<dim_matrix; i++) {
        for(int j=0; j<dim_matrix; j++) {
            mat_a[i*dim_matrix + j] = rand() % MAX;
            mat_b[i*dim_matrix + j] = rand() % MAX;
        }        
    }

    #ifdef DEBUG
    printf("--- MATRICE A ---\n");
    stampa_matrice(mat_a, dim_matrix);
    printf("--- MATRICE B ---\n");
    stampa_matrice(mat_b, dim_matrix);
    printf("--- MATRICE C ---\n");
    stampa_matrice(mat_c, dim_matrix);
    #endif

    double elapsed_parallel = parallel_gemm(mat_a, mat_b, mat_c, dim_matrix);

    #ifdef DEBUG
    printf("--- MATRICE C ---\n");
    stampa_matrice(mat_c, dim_matrix);
    #endif

    printf("Elapsed parallel:\t %f ms;\tSpeedup: %0.2f\n", elapsed_parallel*1000);
}