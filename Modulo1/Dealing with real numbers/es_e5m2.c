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
                // qua c'è dell'errore anche se sto dividendo per una potenza di due
                result += 1/(float)denominatore;
            }

            denominatore = 1;
        }

        if(sign_bit)
            result*=-1;

        return result; 
    }




    // caso standard 
    int denominatore = 1;
    int esponente = 0;
    float valore_esponente;
    result = 1.0; //valore sottinteso

    // qua devo partire dal fondo
    for(int i=ESPONENTE_LENGTH-1; i>=0; i--) {
        if(exp_bits[i]) {
            esponente += (1 << ESPONENTE_LENGTH-(i+1)); 
        }
    }

    // polarizziamo l'esponente 
    esponente -= 15;
    printf("\tesponente: %d\n", esponente);
    
    //calcolo il valore corrispondente
    if(esponente >= 0 ) {
        valore_esponente = 1<<esponente;
        printf("\tvalore esponente: %f\n", valore_esponente);
    }
    else {
        int temp = esponente * -1;
        denominatore <<= temp;   
        valore_esponente = 1/(float)denominatore;
        // qua c'è dell'errore anche se sto dividendo per una potenza di due
        printf("\tvalore esponente: 1/%d = %.014f\n", denominatore, valore_esponente);  
    }
    
    denominatore = 1;
    for(int i=0; i<MANTISSA_LENGTH; i++) {
        if(mantissa_bits[i]) {
            denominatore <<= i+1;
            // qua c'è dell'errore anche se sto dividendo per una potenza di due
            result += 1/(float)denominatore;
        }

        denominatore = 1;
    }
    printf("\tmantissa: %f\n", result);

    result *= valore_esponente;
    if(sign_bit)
        result*=-1;

    return result; 
}

int main() {
    char bits[LENGTH] = {1, 1, 1, 1, 0, 1, 0, 1};
    printf("bits: %d%d%d%d%d%d%d%d\n", bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7]);

    float value = compute_E5M2_value(bits);

    printf("numerical value = %0.14f\n", value);

    return 0;
}