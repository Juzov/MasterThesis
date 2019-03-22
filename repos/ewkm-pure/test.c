#include <stdlib.h>
#include <stdio.h>

double unif_rand(){
	return (double)rand() / (double)RAND_MAX;
}

int main(){
	srand(123456);
	printf("unif_rand %f",unif_rand());
	printf("unif_rand %f",unif_rand());

}
