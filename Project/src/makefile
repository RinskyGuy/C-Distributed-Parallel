build:
	mpicc -fopenmp -c Project.c -o Project.o
	mpicc -fopenmp -o proj Project.o
clean:
	rm -f *.o ./proj
run:
	mpiexec -np 5 ./proj
run2pc:
	mpiexec -np 3 --machinefile HostsIP --map-by node ./proj
