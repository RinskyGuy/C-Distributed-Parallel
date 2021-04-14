/*
 ============================================================================
 Name        : Main.c
 Author      : Guy Rinsky
 ID          : 205701907
 ============================================================================
 */
#include <time.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "ErrorManagement.h"

#define FILEINPUT "input.txt"
#define FILEOUTPUT "output.txt"
#define NS1 3000
#define NS2 2000
#define COLONS_LENGTH 9
#define COLONS (const char*[COLONS_LENGTH]) {"NDEQ","NEQK","STA","MILV","QHRK","NHQK","FYW","HY","MILF"}
#define DOT_LENGTH 11
#define DOTS (const char*[DOT_LENGTH]) {"SAG","ATV","CSA","SGND","STPA","STNK","NEQHRK","NDEQHK","SNDEQK","HFY","FVLIM"}
#define WORK_TAG 1
#define FINISH_TAG 0
#define MASTER_RANK 0

//This struct is like an ID for each thread\process,
//it holds the sentence number, mutant number and offset number required to do the specific task
typedef struct tasksPool {
	int sentence;
	int mutant;
	int offset;
} tasksPool;

void pragmafunction(tasksPool *tasks,tasksPool *taskRes,char *constLine, char **linesForComparisons,char* argv);
void freePointers(char *constLine, char **linesForComparisons, int numOfLines);
void printToFile(int *sol, int numOfLines, char *argv);
double get_Score(const char *constLine, char *mutantLine, int offset);
void readFromFile(char *constLine, char ***linesForComparisons, int *numOfLines,char *argv);
int dotGroupChecker(char x, char y);
int colonGroupChecker(char x, char y);
void get_MutantedSentence(char **mutantedSentence, char *originalSentence,int pos);
void init_TasksPool(tasksPool *tasks);
void createTaskType(MPI_Datatype *MPI_TASK);

double weights[4] = { 0, 0, 0, 0 }; //Global variable of the weights of each comparison (dot,spaces,colons and asterixs)

//This function reads the variables and sentences from the input file
void readFromFile(char *constLine, char ***linesForComparisons, int *numOfLines,char *argv) {
	FILE *fp;
	check_error(((fp = fopen(FILEINPUT, "r")) == NULL ? -1 : 1), argv,FILE_OPEN_ERR);
	int i;
	if (fscanf(fp, "%lf %lf %lf %lf\n", &weights[0], &weights[1], &weights[2],&weights[3]) != EOF) {
		if (fscanf(fp, "%s\n", constLine) != EOF) {
			if (fscanf(fp, "%d\n", numOfLines) != EOF) {
				check_error(((*linesForComparisons = (char**) realloc(*linesForComparisons,(*numOfLines) * sizeof(char*))) == NULL ? -1 : 1),argv, MALLOC_ERR);
				for (i = 0; i < *numOfLines; i++) {
					check_error((((*linesForComparisons)[i] = (char*)malloc(NS2 * sizeof(char))) == NULL ? -1 : 1),argv, MALLOC_ERR);
					if (fscanf(fp, "%s\n", (*linesForComparisons)[i]) == EOF) {
						break;
					}
				}
			}
		}
	}
	check_error((fclose(fp) == EOF ? -1 : 1), argv, FILE_CLOSE_ERR);
}

//This function checks what result it gets on each comparison
//between two chars(until it checks the whole string) and returns the score
double get_Score(const char *constLine, char *mutantLine, int offset) {
	int i, astrixes = 0, colons = 0, dots = 0, spaces = 0;
	for (i = 0; i < strlen(mutantLine); i++) {
		if (constLine[i + (offset)] == mutantLine[i]) {
			astrixes++;
		} else {
			if (colonGroupChecker(constLine[i + (offset)], mutantLine[i])) {
				colons++;
			} else {
				if (dotGroupChecker(constLine[i + (offset)], mutantLine[i])) {
					dots++;
				} else
					spaces++;
			}
		}
	}
	return astrixes * weights[0] - colons * weights[1] - dots * weights[2]
			- spaces * weights[3];
}

//Check if the comparison is a colon
int colonGroupChecker(char x, char y) {
	int i;
	for (i = 0; i < COLONS_LENGTH; i++) {
		if (strchr(COLONS [i], x) != NULL && strchr(COLONS [i], y) != NULL)
			return 1;
	}
	return 0;
}

//Check if the comparison is a dot
int dotGroupChecker(char x, char y) {
	int i;
	for (i = 0; i < DOT_LENGTH; i++) {
		if (strchr(DOTS [i], x) != NULL && strchr(DOTS [i], y) != NULL)
			return 1;
	}
	return 0;
}

//This function runs on the string and appends a hypen at the pos (variable)
//location and completes the entire string
void get_MutantedSentence(char **mutantedSentence, char *originalSentence,int pos) {
	int i;
	for (i = 0; i < strlen(originalSentence) + 2; i++) {
		if (i == pos)
			(*mutantedSentence)[i] = '-';
		else if (i < pos) {
			(*mutantedSentence)[i] = originalSentence[i];
		} else {
			(*mutantedSentence)[i] = originalSentence[i - 1];
		}
	}
	(*mutantedSentence)[strlen(originalSentence) + 1] = '\0';
}

//Initialize the struct
void init_TasksPool(tasksPool *tasks) {
	tasks->sentence = 1;
	tasks->mutant = -1;
	tasks->offset = -1;
}

//Create a derived datatype for sending through MPI - creates datatype tasksPool
void createTaskType(MPI_Datatype *MPI_TASK) {
	tasksPool task;
	int blocklen[3] = { 1, 1, 1 };
	MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_INT };
	MPI_Aint displacements[3];
	displacements[0] = (char*) &task.sentence - (char*) &task;
	displacements[1] = (char*) &task.mutant - (char*) &task;
	displacements[2] = (char*) &task.offset - (char*) &task;
	MPI_Type_create_struct(3, blocklen, displacements, types, MPI_TASK);
	MPI_Type_commit(MPI_TASK);
}

//Print the solution to a file
void printToFile(int *sol, int numOfLines, char *argv) {
	int i;
	FILE *fp;
	check_error(((fp = fopen(FILEOUTPUT, "w")) == NULL ? -1 : 1), argv,FILE_OPEN_ERR);
	for (i = 1; i <= numOfLines; i++) {
		check_error(fprintf(fp, "n = Offset = %d, k = MS(%d)\n", sol[2 * i - 2],sol[2 * i - 1]), argv, FILE_PRINT_ERR);
	}
}

//This function free's allocated memory when it is no longer needed
void freePointers(char *constLine, char **linesForComparisons, int numOfLines) {
	int j;
	free(constLine);
	for (j = 0; j < numOfLines; j++) {
		free(linesForComparisons[j]);
	}
	free(linesForComparisons);
}

//This function opens the threads and sends data to another function to compute the score for each individual thread/
//This function is meant to run whether on 1 process or multiple processes.
void pragmafunction(tasksPool *tasks,tasksPool *taskRes,char *constLine, char **linesForComparisons,char* argv){
char *mutantSqn; //Pointer to the mutant string
double score = 0; //The score each thread will have adjust accordingly to it's mutant
int k;
double maxScore = - DBL_MAX; //Assign the lowest number possible for double
int numOfMutants = strlen(linesForComparisons[tasks->sentence-1]);
#pragma omp parallel num_threads(numOfMutants) shared(maxScore, taskRes) private(mutantSqn) firstprivate(tasks, score)
			{
				//Private variable for every thread to hold it's specific mutant
				check_error(((mutantSqn = (char*) malloc((strlen(linesForComparisons[tasks->sentence- 1]) + 2) * sizeof(char)))== NULL ? -1 : 1), argv, MALLOC_ERR);

				//Every thread does one 'k' index
				#pragma omp for
				for (k = 1; k <= numOfMutants; k++) {
					tasks->mutant = k; //Assign the mutant number to the specific thread
					get_MutantedSentence(&mutantSqn,linesForComparisons[tasks->sentence - 1], k); //make the mutant

					//Every thread does all 'j' index's
					for (int j = 0;j < (strlen(constLine) - strlen(mutantSqn)) + 1;j++) {
						score = get_Score(constLine, mutantSqn, j);

						//Only 1 thread can enter at a time
						#pragma omp critical
						{
							if (score > maxScore) {
								tasks->offset = j; //Assign new maximum offset to specific task at a specific thread
								tasks->mutant = k;//Assign new maximum mutant to specific task at a specific thread
								maxScore = score; //Assign new maximum score to shared variable
								*taskRes = *tasks; //Assign new task details with maximum score to shared variable
							}
						}
					}
					free(mutantSqn); //Free the allocation in each thread after mutantSqn is no longer needed
				}

			}
}

int main(int argc, char *argv[]) {
	clock_t start, end;
	MPI_Datatype MPI_TASK; //Datatype of struct 'tasksPool'
	MPI_Status status; //Return status for receive

	tasksPool tasks, taskRes; //2 tasksPool type for sending the specific task and receiving the answer for the specific task
	char *constLine, **linesForComparisons; //2 strings pointers, first one to hold the original sequence and the second one holds all the sequences we want to compare with the original
	int i, numOfLines = 0; //3 'for' variables and a variable to indicate how much line sequences we have to compute
	int workingProcesses = 0; //Indicates how many processes are active
	int my_rank; //Rank of process
	int numOfProcesses; //Number of processes

	MPI_Init(&argc, &argv); //Start up MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //Find out process rank
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses); //Find out number of processes

	check_error(((constLine = (char*) malloc(NS1 * sizeof(char))) == NULL ? -1 : 1),argv[0], MALLOC_ERR);
	check_error(((linesForComparisons = (char**) malloc(sizeof(char*))) == NULL ? -1 : 1), argv[0], MALLOC_ERR);
	createTaskType(&MPI_TASK);
	readFromFile(constLine, &linesForComparisons, &numOfLines, argv[0]);

	//In case of multiple processes
	if (numOfProcesses > 1) {
		if (my_rank == 0) { //Master's code
			start = clock(); //Start timer
			init_TasksPool(&tasks);
			int *sol; //1D array to hold the solutions for printing in a specific order
			check_error(((sol = (int*) malloc(2 * numOfLines * sizeof(int))) == NULL ?-1 : 1), argv[0], MALLOC_ERR);

			//Picks whether to iterate numOfProcesses or numOfLines times, to avoid error in sending to the right rank
			for (i = 1;i< ((numOfProcesses <= numOfLines) ?numOfProcesses : numOfLines + 1); i++) {
				MPI_Send(&tasks, 1, MPI_TASK, i, WORK_TAG, MPI_COMM_WORLD); //Send tasks to all the processes
				tasks.sentence++; //Next sentence (task) to send
				workingProcesses++; //+1 more process is active now.
			}
			do {
				MPI_Recv(&taskRes, 1, MPI_TASK, MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, &status); //Recieve a specific task
				workingProcesses--; //1 less process is active now

				sol[2 * taskRes.sentence - 2] = taskRes.offset; //Both of this lines put results in a special format order
				sol[2 * taskRes.sentence - 1] = taskRes.mutant;

				if (tasks.sentence == numOfLines + 1) { //Checks if all tasks were processed, if so it will send finish tags to the processes
					MPI_Send(&tasks, 1, MPI_TASK, status.MPI_SOURCE, FINISH_TAG,MPI_COMM_WORLD);

				} else {
					MPI_Send(&tasks, 1, MPI_TASK, status.MPI_SOURCE, WORK_TAG,MPI_COMM_WORLD); //Sends the last answering process a new task
					tasks.sentence++; //Next sentence (task) to send
					workingProcesses++; //+1 more process is active now.
				}

				//Master will send more tasks until there are no more active processes (which will be terminated if there are no more tasks)
			} while (workingProcesses > 0);
			printToFile(sol, numOfLines, argv[0]);
			free(sol); //Free allocated memory when it is no longer needed
		}

		else if (my_rank != 0 && my_rank <= numOfLines) { //Slave's code
			while (1) {
				MPI_Recv(&tasks, 1, MPI_TASK, MASTER_RANK, MPI_ANY_TAG,MPI_COMM_WORLD, &status);

				//The master process will send a termination tag the slave process is no longer needed and can be terminated.
				if (status.MPI_TAG == FINISH_TAG) {
					break;
				}
				pragmafunction(&tasks,&taskRes,constLine,linesForComparisons,argv[0]);
				MPI_Send(&taskRes, 1, MPI_TASK, MASTER_RANK, WORK_TAG,MPI_COMM_WORLD); //Sends the answer back to the master
			}

		}
	}
	//In case of one process
	else {
		start = clock();//Start timer
		init_TasksPool(&tasks);
		int solArray[2*numOfLines];//Array for holding the solutions in a specific order to wright afterwards at the output file

		//This 'for' iterates over all the lines and computes its best scores
		for (i=1;i<=numOfLines;i++,tasks.sentence++){
			pragmafunction(&tasks,&taskRes,constLine,linesForComparisons,argv[0]);
			solArray[2 * tasks.sentence - 2] = taskRes.offset;//Both of this lines hold the solutions in a specific order
			solArray[2 * tasks.sentence - 1] = taskRes.mutant;
		}
		printToFile(solArray, numOfLines, argv[0]);
	}

	freePointers(constLine, linesForComparisons, numOfLines);
	MPI_Finalize();	//Shut down MPI
	if (my_rank == 0) {
		end = clock();	//Stop timer
		printf("\ntime taken = %lf", (((double) end - start) / CLOCKS_PER_SEC));//Adjust the timer and print how much time is taken
	}
	return 0;
}
