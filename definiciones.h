#ifndef DEF_H
#define DEF_H

#include "stdlib.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>


#define NUM_CASILLAS 120
#define CAMBIO_LADO 6 
#define MAXFEN 128
#define MAXGAMEMOVES 2048
#define PROFUNDIDAD 100
#define TIEMPO_MAX 2500000
#define FCCAS(col,fila) ( (col + (21) ) + ( (fila) * 10 ) ) 
/***********************************************************/
/* MACRO: START_FEN                             
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*
/* Descripción:
/* FEN a la posición inicial del tablero. Esencial para empezar una partida
/***********************************************************/

#define START_FEN  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

/***********************************************************/
/* Array: PieceVal                             
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*
/* Descripción:
/* Array que le da un valor material a cada pieza. Cada posición corresponde a PieceVal[pieza]
/***********************************************************/

__constant__ int device_PieceVal[13]= { 0, 100, 325, 325, 550, 1000, 50000, 100, 325, 325, 550, 1000, 50000  };
const int host_PieceVal[13]= { 0, 100, 325, 325, 550, 1000, 50000, 100, 325, 325, 550, 1000, 50000  };

extern __device__ __host__ int PieceVal(int i);

/***********************************************************/
/* Enumeraciones                             
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*
/* Descripción:
/* Declaramos varias enumeraciones que nos serán útiles en el desarrollo del módulo. La primera enumeración es una
/* forma de codificar las distintas piezas en el ajedrez (white or black, Pawn, Knight (N), Bishop, Rook, Queen, King).
/* Los dos siguientes son una forma de codificar las distintas columnas (de A a H) y filas (de 1 a 8) que hay en el tablero.
/* Además tenemos una enumeración para codificar el lado al que le toca. Además, tenemos otra enumeración que codifica las casillas 
/* que hay en el tablero.(Las casillas tienen los números de los índices de la estructura Tablero).
/* Además, hay otras enumeraciones de utilidad en las fnciones como OK, ERR, TRUE , FALSE o GANAN_NEGRAS, TABLAS, GANAN_BLANCAS.
/* Por últimos tenemos la forma de codificar los distintos enroques para llevar cuenta del permiso de enroque.
/***********************************************************/


enum { EMPTY, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK  };
enum { COL_A, COL_B, COL_C, COL_D, COL_E, COL_F, COL_G, COL_H, COL_NONE };
enum { FILA_1, FILA_2, FILA_3, FILA_4, FILA_5, FILA_6, FILA_7, FILA_8, FILA_NONE };

enum { WHITE, BLACK, BOTH };

enum {
  A1 = 21, B1, C1, D1, E1, F1, G1, H1,
  A2 = 31, B2, C2, D2, E2, F2, G2, H2,
  A3 = 41, B3, C3, D3, E3, F3, G3, H3,
  A4 = 51, B4, C4, D4, E4, F4, G4, H4,
  A5 = 61, B5, C5, D5, E5, F5, G5, H5,
  A6 = 71, B6, C6, D6, E6, F6, G6, H6,
  A7 = 81, B7, C7, D7, E7, F7, G7, H7,
  A8 = 91, B8, C8, D8, E8, F8, G8, H8, NO_SQ, OFFBOARD
};

enum { FALSE, TRUE };
enum {ERR=-1,OK=1};
enum {GANAN_NEGRAS = 2, TABLAS, GANAN_BLANCAS,EXIT,MOD};


enum { WKCA = 1, WQCA = 2, BKCA = 4, BQCA = 8 };


/***********************************************************/
/* Estructura: MOVE                             
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*
/* Descripción:
/* Estructura que guarda los distintos movimientos que se hacen. Tiene los parámetros castle (para controlar si el movimiento ha 
/* sido de enroque), from que muestra la casilla de salida, to, que muestra la casilla a la que se mueve la pieza, una array de piezas
/* y, por último, si se ha hecho una captura al paso
/***********************************************************/


typedef struct{
	int castle;
	int from;
	int to;
	//el primer elemento es la pieza que se ha movido, el segundo lo que ha capturado, y lo último en que se ha coronado
	int piezas[3];
	int paso;
}MOVE;

/***********************************************************/
/* Estructura: S_UNDO                             
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*
/* Descripción:
/* Estructura que guarda la información necesaria para volver atrás jugadas. Guarda la jugada que se ha hecho, como estaban los permisos de enroque,
/* La casilla en la que se puede comer al paso, el número de 50 jugadas, y una fen que guarda el estasdo de la posición.
/***********************************************************/

typedef struct {

	MOVE * jugada;
	int enroque;
	int AlPaso;/*casilla en la que se puede comer al paso*/
	int fiftyMove;
	char *fen;

} S_UNDO;

/***********************************************************/
/* Estructura: TABLERO                           
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*
/* Descripción:
/* Estructura que guarda la información del tablero. En ella podemos ver varias partes. La primera parte consta de
/* el array pieces, un array de tamaño 120 que guarda un tablero como vemos en la figura:
/*		________________________________________
/*		|_0_|_1_|___|___|___|___|___|___|___|_9_|
/*		|_10|___|___|___|___|___|___|___|___|___|
/*		|_20|_A1|_B1|_C1|_D1|_E1|_F1|_G1|_H1|___|
/*		|_30|_A2|_B2|___|_._|_._|_._|___|_H2|___|
/*		|_40|_A3|_B3|___|___|___|___|___|_H3|___|
/*		|_50|_A4|_B4|___|_._|_._|_._|___|_H4|___|
/*		|_60|_A5|_B5|___|___|___|___|___|_H5|___|
/*		|_70|_A6|_B6|___|_._|_._|_._|___|_H6|___|
/*		|_80|_A7|_B7|___|___|___|___|___|_H7|___|
/*		|_90|_A8|_B8|_C8|_D8|_E8|_F8|_G8|_H8|___|
/*		|100|101|___|___|___|___|___|___|___|109|
/*		|110|111|___|___|___|___|___|___|___|119|
/*
/*El array guarda enteros que son las codificaciones de las piezas. Como el tablero de ajedrez en realidad tiene 64 casillas,
/*las casillas que están fuera del tablero se rellenan como OFFBOARD. EL array de 120 nos permite un desarrollo más sencillo de la
/* función de generar movimientos.
/*
/*Además de la estructura de pieces, tenemos un array donde guardamos las casillas de los reyes, que nos permiten controlar fácilmente cuando una 
/* jugada es jaque. Tenemos un entero sobre a quién el toca jugar (WHITE o BLACK), un entero que guarda la casilla por la que se podría comer al paso
/* (o EMPTY/NO_SQ) si no hay ninguna. Tenemos también un contador de la regla de las 50 jugada, para controlar los casos de tablas, un entero que muestra el
/* número de jugadas (se incrementan en uno cada vez que alguien juega, no exactamente como en el juego normal) y un entero en el que están codificados los permisos de enroque.
/* Hay también un array que guarda el número de piezas de cada tipo que hay en el tablero (pieces), Un array que guarda el material de la posición (definido en pieceVal),
/* un array de estructuras UNDO que nos permite volver atrás en las jugadas junto a un entero histcont que muestra el número de elementos en S_UNDO. Finalmente
/* tenemos un array bidimensional que guarda las casillas en las que se encuentran cada una de las piezas que hay de cada tipo.
/***********************************************************/


typedef struct {

	int *pieces;//tablero en si

		
	int *KingSq;
	
	int side;
	int AlPaso;
	int fiftyMove;

	int histcont; //numero de elementos de history
	int j_real;//numero de jugadas
	
	int enroque;
	

	
	int *pceNum;//numero de piezas de cada tipo
	int *material;
	
	S_UNDO **history;
	
	// piece list
	int **pList;//lista por cada tipo de pieza guarda su casilla
	
} TABLERO;

/***********************************************************/
/* Estructura: TABLERO                           
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*
/* Descripción:
/* Esta estructura es utilizada para la busqueda, en la función AlphaBeta. Contiene un entero, visited que indica el número
/* de nodos visitados, otro entero, bestScore, que contiene la puntuación del mejor movimiento encontrado, y un último entero, depth,
/* que indica hasta la profundidad hasta la que se debe de llegar
/*
/***********************************************************/

typedef struct{
	int visited;//numero de nodos visitados 
	int bestScore;
	int depth;//profundidad a la que se enceuntra
	double tiempo;
	double maxtemp;
	int stop;
	

}INFO;


__constant__ int device_FILAsBrd[NUM_CASILLAS] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 2, 2, 2, 2, 2, 2, 2, 2, 100, 100, 3, 3, 3, 3, 3, 3, 3, 3, 100, 100, 4, 4, 4, 4, 4, 4, 4, 4, 100, 100, 5, 5, 5, 5, 5, 5, 5, 5, 100, 100, 6, 6, 6, 6, 6, 6, 6, 6, 100, 100, 7, 7, 7, 7, 7, 7, 7, 7, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
__constant__ int device_COLsBrd[NUM_CASILLAS] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
const int host_FILAsBrd[NUM_CASILLAS] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 2, 2, 2, 2, 2, 2, 2, 2, 100, 100, 3, 3, 3, 3, 3, 3, 3, 3, 100, 100, 4, 4, 4, 4, 4, 4, 4, 4, 100, 100, 5, 5, 5, 5, 5, 5, 5, 5, 100, 100, 6, 6, 6, 6, 6, 6, 6, 6, 100, 100, 7, 7, 7, 7, 7, 7, 7, 7, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
const int host_COLsBrd[NUM_CASILLAS] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};


/* FUNCTIONS */

// init.c
extern void InitFILAsCOLsBrd();


// tablero.c
extern __host__ __device__ int pieceColour(int pce);
extern void ResetBoard(TABLERO *pos);
extern int LeerFen(char *fen, TABLERO *pos);
extern __host__ __device__ void PrintBoard(const TABLERO *pos);
extern __host__ __device__ void UpdateListsMaterial(TABLERO *pos);
extern __host__ __device__ int CheckBoard(const TABLERO *pos);
extern __host__ __device__ int C120a64(int c120);
extern __host__ __device__ int C64a120(int c64);
extern __host__ __device__ void Free_tablero(TABLERO *tab);
extern __host__ __device__ TABLERO* Create_tablero();
int __host__ __device__ Cas_Col (int cas);
int __host__ __device__ Cas_Fila (int cas);
extern __host__ __device__ char *EscribirFen(TABLERO *t);
int esTablas(TABLERO *tab);
int Repetida(TABLERO *tab, int *times);
int InsufMat(TABLERO *tab);
int FinPartida(TABLERO *tab);


// ataque.c
extern __host__ __device__ int SqAttacked(const int sq, const int side, const TABLERO *pos);

//movimientos.c
extern __device__ __host__ int COLsBrd(int i);

extern __device__ __host__ int FILAsBrd(int i);




extern __host__ __device__ MOVE **Generador_Peones(TABLERO *t, MOVE **m, int *count );
extern __host__ __device__ MOVE **Generador_Movimientos(TABLERO *t, int *count);
extern __host__ __device__ MOVE** Generador_RC(TABLERO *t, MOVE **m, int *count);
int print_moves(MOVE **m, int count);
extern __host__ __device__ MOVE ** Generador_Slide(TABLERO *t, MOVE **m, int *count );
extern __host__ __device__ MOVE ** Generador_Enroques(TABLERO *t, MOVE **m, int *count );
extern int PrintMove(MOVE *mt);
extern __host__ __device__ int HacerJugada(TABLERO *t,MOVE *m);
extern void DeshacerJugada(TABLERO *pos);
extern __host__ __device__ void free_UNDO(S_UNDO * u);
extern __host__ __device__ S_UNDO *create_UNDO (MOVE *jugada);
extern __host__ __device__ MOVE *move_copy(MOVE*m);
extern __host__ __device__ MOVE *create_move();
extern __host__ __device__ void free_move(MOVE *m);
extern __host__ __device__ MOVE* insert_move(int castle, int from, int to, int pieza, int captura, int corona, int paso);
int move_cmp(MOVE *m1, MOVE *m2);
//comprobacion.c

void Comprobacion(int prof, TABLERO *pos);
void Comprobaciontest(int prof, TABLERO *pos);

//interfaz.c
int is_Valid(MOVE *m,TABLERO *t);
MOVE *LeerMovimiento(char *entrada, TABLERO *t);
int Menu_juego(TABLERO *tab);

//evaluacion.c
int EvalPosition(const TABLERO *pos);
int Mirror64(int sq64);
//busqueda.c
MOVE* SearchPosition(TABLERO *pos, INFO  *info);
static int AlphaBeta(int alpha, int beta, int depth, TABLERO *pos, INFO *info,MOVE** Best);

#endif