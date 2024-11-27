#include <stdio.h>
#include <stdlib.h>

#define LENGTH 8
#define MANTISSA_LENGTH 2
#define ESPONENTE_LENGTH 5

float compute_E5M2_value(char* bits) {
    float result = 0.0;

    char sign_bit        = bits[0];
    char* exp_bits       = bits + 1;
    char* mantissa_bits  = bits + 6;

    // esponente con tutti i bit a uno 
    if( exp_bits[0] && exp_bits[1] && exp_bits[2] && exp_bits[3] && exp_bits[4] ) {
        // mantissa con tutti i bit a zero
        if( !mantissa_bits[0] && !mantissa_bits[1] ) {
            // +/- inf
            sign_bit ? printf("-inf!\n") : printf("+inf!\n");
            exit(0);
        }
        else {
            // nan
            printf("NaN\n");
            exit(0);
        }
    }

    // esponente con tutti i bit a zero
    if( !exp_bits[0] && !exp_bits[1] && !exp_bits[2] && !exp_bits[3] && !exp_bits[4] ) {
        // valore subnormale
        int denominatore = 1;

        for(int i=0; i<MANTISSA_LENGTH; i++) {
            if(mantissa_bits[i]) {
                denominatore <<= i+1;
                result += 1/(float)denominatore;
            }

            denominatore = 1;
        }

        if(sign_bit)
            result*=-1;

        return result; 
    }

    // caso standard 
    result = 1.0;
    
    int denominatore = 1;
    int termine_esponenziale = 0;

    for(int i=0; i<ESPONENTE_LENGTH; i++) {
        if(exp_bits[i]) {
            termine_esponenziale += (2 << (ESPONENTE_LENGTH-i+1)); 
        }
    }

    // polarizziamo l'esponente 
    termine_esponenziale -= 15;
    printf("\ttermine_esponenziale: %d\n", termine_esponenziale);

    for(int i=0; i<MANTISSA_LENGTH; i++) {
        if(mantissa_bits[i]) {
            denominatore <<= i+1;
            result += 1/(float)denominatore * termine_esponenziale;
        }

        denominatore = 1;
    }

    return result; 
}

int main() {
    char bits[LENGTH] = {0, 0, 0, 0, 0, 1, 0, 1};

    float value = compute_E5M2_value(bits);

    printf("bits: %d%d%d%d%d%d%d%d\nnumerical value = %f\n", bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7], value);

    return 0;
}