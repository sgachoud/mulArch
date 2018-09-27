/*
============================================================================
Filename    : function.c
Author      : Sébastien Gachoud
SCIPER		: 250083
============================================================================
*/

#include <math.h>

double identity_f (double x){
    return x;
}

double square_f (double x){
    return x*x;
}

double poly_f (double x){
    return 5*x*x*x+10*square_f(x)+29*x+450;
}