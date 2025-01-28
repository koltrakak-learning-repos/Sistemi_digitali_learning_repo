#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "omp.h"

#define MAX  5
// #define DEBUG 1

void stampa_matrice( int dim, int mat[][dim]) {
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            printf("%d\t", mat[i][j]);
        }     

        printf("\n");   
    }

    printf("\n");
}

double gemm(int dim, int mat_a[][dim], int mat_b[][dim], int mat_c[][dim]) {
    double start = omp_get_wtime();

    // i primi due cicli considerano ogni elemento della matrice
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            // il singolo elemento della matrice risultante viene calcolato considerando un intera riga/colonna
            for(int k=0; k<dim; k++) {
                mat_c[i][j] += mat_a[i][k] * mat_b[k][j];
            }  
        }     
    }

    double end = omp_get_wtime();

    return end-start;
}

double parallel_gemm(int dim, int mat_a[][dim], int mat_b[][dim], int mat_c[][dim]) {
    double start = omp_get_wtime();

    #pragma omp parallel for
    // i primi due cicli considerano ogni elemento della matrice
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            // il singolo elemento della matrice risultante viene calcolato considerando un intera riga/colonna
            for(int k=0; k<dim; k++) {
                mat_c[i][j] += mat_a[i][k] * mat_b[k][j];
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
    
    omp_set_num_threads(num_threads);
    printf("Numero massimo di thread eseguibili in parallelo: %d\n", omp_get_max_threads());

    int mat_a[dim_matrix][dim_matrix];
    int mat_b[dim_matrix][dim_matrix];
    int mat_c[dim_matrix][dim_matrix];
    memset(mat_c, 0, sizeof(mat_c));

    srand(time(NULL));
    for(int i=0; i<dim_matrix; i++) {
        for(int j=0; j<dim_matrix; j++) {
            mat_a[i][j] = rand() % MAX;
            mat_b[i][j] = rand() % MAX;
        }        
    }

    #ifdef DEBUG
    printf("--- MATRICE A ---\n");
    stampa_matrice(dim_matrix, mat_a);
    printf("--- MATRICE B ---\n");
    stampa_matrice(dim_matrix, mat_b);
    #endif

    double elapsed_sequential = gemm(dim_matrix, mat_a, mat_b, mat_c);
    double elapsed_parallel = parallel_gemm(dim_matrix, mat_a, mat_b, mat_c);

    #ifdef DEBUG
    printf("--- MATRICE C ---\n");
    stampa_matrice(dim_matrix, mat_c);
    #endif

    printf("Elapsed sequential:\t %f ms\n", elapsed_sequential*1000);
    printf("Elapsed parallel:\t %f ms;\tSpeedup: %0.2f\n", elapsed_parallel*1000, elapsed_sequential/elapsed_parallel);
}