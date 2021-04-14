#ifndef ERRORMANAGEMENT_H_
#define ERRORMANAGEMENT_H_

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>


#define _ERROR_MSG "%s: %s Error! errono: %d\n"

#define MALLOC_ERR			0
#define FILE_OPEN_ERR		1
#define FILE_CLOSE_ERR		2
#define FILE_PRINT_ERR		3

//This function checks errors and if it finds an error
//it will print to stderr, abort the MPI and terminate the program
void check_error(int err, const char* name, char reason){
	const char* MSG[] = {
			"Malloc",
			"File did not open",
			"File did not close",
			"File did not print properly"
	};
	if(err >= 0)
		return;
	fprintf(stderr,_ERROR_MSG, name, MSG[(int)reason], errno);
	MPI_Abort(MPI_COMM_WORLD,__LINE__);
	exit(err);
}


#endif /* ERRORMANAGEMENT_H_ */


