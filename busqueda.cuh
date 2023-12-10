#include "stdio.h"
#include "definiciones.h"
#include <time.h>
#include "parallel.cuh"

#define INFINITO 50000
#define JAQUEMATE 30000
#define PROFMAX 64
#define NOMOV 0

/***********************************************************/
/* Función: Alphabeta                             
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*                                                         
/* Parámetros de entrada:
/* alpha: mejor opcion para el maximizador
/* beta: mejor opcion para el minimizador
/* depth: profundidad del algoritmo                                                  
/* pos: puntero a un tablero
/* info: puntero a tipo de dato info con los datos de la busqueda
/* best: puntero donde se almacenara el movimiento seleccionado
/* 
/* Retorno:
/* alfa: alfa/beta en cada caso
/*
/* Descripción:
/* Algoritmo de busqueda del mejor movimiento a realizar a una cierta profundidad seleccionada
/*
/* Mas en Detalle:
/* Implementa el algoritmo de poda alfabeta, cuyo proposito es encontrar el mejor movimiento. 
/* Descarta las ramas del arbol que en función de los valores de alfa y beta no hace falta explorar
/***********************************************************/

static int AlphaBeta(int alpha, int beta, int depth, TABLERO *pos, INFO *info,MOVE** Best, MOVE* arbol_jugadas, int num_jugadas, int arbol_depth) { 
	int Legal = 0;
	int Score = -INFINITO;
	MOVE * movelist;
	int count,c;
	int index=0;
	double tiempo;
	clock_t c1, c2,c3,c4;
    int *acc_counts;
    MOVE *arbol;

    printf("Entro en alfabeta, depth= %d\n",depth);
	//ASSERT(CheckBoard(pos)); 

	if(info->stop==TRUE){
		return 0;
	}
	

	if(info->tiempo >=info->maxtemp){
			info->stop=TRUE;
			return 0;
	}


	c1 = clock();         /*Clock 1*/

	if(depth == 0) {
		info->visited++;
		
		return EvalPosition(pos);
	}
	info->visited++;
	
	
	if(esTablas(pos)) {
		return 0;
	}
	
	Score = -INFINITO;
	
  if (arbol_depth == 0){
    movelist = Generador_Movimientos_GPU(pos,&count, &acc_counts, &arbol);
    printf("JUGADAAAAA3, from: %d, to %d, piece %d\n", movelist[2].from, movelist[2].to, movelist[2].piezas[0]); 
  }
  else{
    movelist = arbol_jugadas;
    //printf("num jugadas: %d\n",num_jugadas);
    count = num_jugadas;
  }
      
    
	c2 = clock();      /* clock 2*/

   
	/*Actualizacion del tiempo*/
	tiempo = (double)(c2-c1);    
	info->tiempo+= tiempo;

	
	printf("Estamos en alfa beta\n");
	for(index= 0; index< 1; index++) {	


		c3=clock();
       
        if ( HacerJugada(pos,&(movelist[index]))==FALSE)  {
            continue;
        }
    
    PrintMove(&(movelist[3]));
    PrintBoard(pos); 
    printf("adios\n");  
		Legal++;


		c4=clock();

		/*Actualizacion del tiempo*/
		tiempo = (double)(c4-c3); 
		info->tiempo+= tiempo;
		
    if(arbol_depth == 0){
      Score = -AlphaBeta( -beta, -alpha, depth-1, pos, info, Best, &(arbol[acc_counts[index]]), acc_counts[index+1] - acc_counts[index], 1);
    }
    else{
		  Score = -AlphaBeta( -beta, -alpha, depth-1, pos, info, Best, NULL, 0, 0);		
    }
    DeshacerJugada(pos);
		if(Score >= beta) {
			if(arbol_depth == 0){
        free(movelist);
        free(arbol);
      }

			return beta;
		}
		if(Score > alpha) {
			alpha = Score;
			if(depth == info->depth && info->stop==FALSE){  /* IMPORTANTE*/
				free_move(*Best);
				(*Best)=move_copy(&(movelist[index]));

			}
		}	
    }

  if(arbol_depth == 0){
    free(movelist);
    free(arbol);
  }

	if(Legal == 0) {
		if(SqAttacked(pos->KingSq[pos->side],pos->side^1,pos)) {
			
			return -JAQUEMATE+ pos->j_real;
		} else {
			return 0;
		}
	}

	return alpha;
} 


/***********************************************************/
/* Función: SearchPosition                             
/* Autores: Omicron: Pablo Soto, Sergio Leal, Raúl Díaz                                  
/*                                                         
/* Parámetros de entrada:                                            
/* pos: puntero a un tablero
/* info: puntero a tipo de dato info con los datos de la busqueda
/* 
/* Retorno:
/* Puntero al movimiento que se ha seleccionado tras la busqueda
/* 
/* Descripcion:
/* Almacena los datos de la busqueda de alphabeta en info y devuelve el mejor movimiento para una posicion
/* 
/***********************************************************/

MOVE* SearchPosition(TABLERO *pos, INFO  *info) {

	MOVE **Best;
	MOVE *retorno;
	Best=(MOVE**)malloc(sizeof(MOVE*));
	int bestScore = -INFINITO;
	int depth;
	*Best=NULL;

    /*Inicializacion de campos de info*/

	info->tiempo=0;
	info->maxtemp=TIEMPO_MAX;
	info->visited=0;
	info->stop=FALSE;


	info->depth=1;
	bestScore = AlphaBeta(-INFINITO, INFINITO, 1, pos, info,Best,NULL,0,0);
	info->bestScore=bestScore;
	
	retorno=move_copy(*Best);
	free(*Best);


	for(depth=2; depth<PROFUNDIDAD; depth++){
		*Best=NULL;
		info->depth=depth;
		bestScore = AlphaBeta(-INFINITO, INFINITO, depth, pos, info,Best, NULL,0,0);
		
		if (info->stop== FALSE){
			info->bestScore=bestScore;
		
			free(retorno);
			retorno=move_copy(*Best);
			free(*Best);
			PrintMove(retorno);
		}else {
			free(*Best);
			break;
		}
		printf("	Depth: %d\n", depth);
	}

	free(Best);

	return retorno;
}