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

void transpose_matrix(int* mat, int dim) {
    #pragma omp parallel for schedule (static, 1)
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            // attenzione a non scambiare due volte
            if (i > j) {
                #ifdef DEBUG
                printf("scambio [%d][%d] = %d, con [%d][%d] = %d\n", i, j, mat[i*dim + j], j, i, mat[j*dim + i]);
                #endif

                int temp = mat[i*dim + j];
                mat[i*dim + j] = mat[j*dim + i];
                mat[j*dim + i] = temp;
            }
        }     
    }
}


double transposed_gemm(const int* mat_a, int* mat_b, int* mat_c, int dim) {
    double start = omp_get_wtime();
    transpose_matrix(mat_b, dim);
    
    
    #pragma omp parallel for schedule (static, 1)
    // i primi due cicli considerano ogni elemento della matrice
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            // il singolo elemento della matrice risultante viene calcolato considerando un intera riga
            for(int k=0; k<dim; k++) {
                mat_c[i*dim + j] += mat_a[i*dim + k] * mat_b[j*dim + k];
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

    double elapsed_transposed = transposed_gemm(mat_a, mat_b, mat_c, dim_matrix);

    #ifdef DEBUG
    printf("--- MATRICE C ---\n");
    stampa_matrice(mat_c, dim_matrix);
    #endif

    printf("Elapsed transposed:\t %f ms\n", elapsed_transposed*1000);

    free(mat_a);
    free(mat_b);
    free(mat_c);
}