/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the GAIVI machines
   ==================================================================
/* The initial C program was provided by Dr. Tu, however the accelerated version of this program was developed by Krish Shah using CUDA kernels as part of a class project*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#define BOX_SIZE	23000 /* size of the data box on one dimension            */

//each bucket stores a count
typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

bucket * histogram;		/* list of all buckets in the histogram   */
bucket * gpu_histogram; /* this will store the histogram modified by the kernel */

long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */

//for coalesced memory access, struct of arrays, size declared at runtime
typedef struct atom_coordinates {
	double * x_pos;
	double * y_pos;
	double * z_pos;
} atom_array;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;

/* distance of two points in the atom_list */
//using pointers to avoid extra copies
__host__ double p2p_distance(int ind1, int ind2, atom_array atom_array1) {
	
	double x1 = atom_array1.x_pos[ind1];
	double x2 = atom_array1.x_pos[ind2];
	double y1 = atom_array1.y_pos[ind1];
	double y2 = atom_array1.y_pos[ind2];
	double z1 = atom_array1.z_pos[ind1];
	double z2 = atom_array1.z_pos[ind2];
		
	return sqrt((x1 - x2) * (x1-x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

__device__ double p2p_distance_gpu(double x1, double y1, double z1, double x2, double y2, double z2) {
	return sqrt((x1 - x2) * (x1-x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

__global__ void PDH_baseline_parallel(int atom_count, int bucket_width, atom_array *d_atom_array, bucket *d_histogram, int block_count, int num_buckets) {
    
	extern __shared__ double shared_mem[]; 
	//using shared memory to hold private copy of histogram and tile
	//the tile is split into three arrays to coalesce access 

	//tile size is block size and thread num is the global thread number
    int tile_size = blockDim.x; 
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	//shared memory must be used in sequence where the end of the previous space determines the address for the next
    double *R_x = shared_mem;
    double *R_y = &R_x[tile_size];
    double *R_z = &R_y[tile_size];
    bucket *local_hist = (bucket *)&R_z[tile_size];

	//initialize local copy to 0, since we don't know which bucket will be written to, all are 0
    if(threadIdx.x == 0) {
        for (int i = 0; i < num_buckets; i++) {
            local_hist[i].d_cnt = 0;
        }     
    }    

	//make sure its initialized before writing happens
    __syncthreads();

	//make sure that valid memory access happens, we don't want to access atoms that don't exist
    if (thread_num < atom_count) {
        double L_x = d_atom_array -> x_pos[thread_num];
        double L_y = d_atom_array -> y_pos[thread_num];
        double L_z = d_atom_array -> z_pos[thread_num];

		//we only need to load the right tile for every block execpt for the last one
		//so it goes to the second to last block, as this is the last block to have a right tile
        if (blockIdx.x < block_count - 1) {
            for (int i = blockIdx.x + 1; i < block_count; i++) {
				//to load the right block we will use the current iteration to calculate an offset
				//the next block is block dim positions away, the threads work together
                int right_index = blockDim.x * i + threadIdx.x;
                if (right_index < atom_count) {
                    R_x[threadIdx.x] = d_atom_array -> x_pos[right_index];
                    R_y[threadIdx.x] = d_atom_array -> y_pos[right_index];
                    R_z[threadIdx.x] = d_atom_array -> z_pos[right_index];
                }
				//we want to wait for the right tile to be loaded, since threads work together they must all reach this point
                __syncthreads();

				//for the situation where the left block is the second to last block and the right block is the last block
				//we have to find the number of threads leftover and only iterate through those
                int valid_elements = min(blockDim.x, atom_count - (blockDim.x * i));
                for (int j = 0; j < valid_elements; j++) {
                    double current_distance = p2p_distance_gpu(L_x, L_y, L_z, R_x[j], R_y[j], R_z[j]);
                    int h_pos = (int)(current_distance / bucket_width);
                    atomicAdd((unsigned long long *)&(local_hist[h_pos].d_cnt), (unsigned long long)1);
                }

				//ensure each thread is on the same iteration, so no thread uses a previous tile
                __syncthreads();
            }
        }

		//interblock computation
        R_x[threadIdx.x] = d_atom_array -> x_pos[thread_num];
        R_y[threadIdx.x] = d_atom_array -> y_pos[thread_num];
        R_z[threadIdx.x] = d_atom_array -> z_pos[thread_num];
		//since threads work together we need them to all reach this point to ensure the block is completely done loading
        __syncthreads();

		//this ensures that inter block computation happens correctly for the last block too
        int valid_intra_block = min(blockDim.x, atom_count - (blockDim.x * blockIdx.x));
        for (int i = threadIdx.x + 1; i < valid_intra_block; i++) {
            double current_distance2 = p2p_distance_gpu(R_x[threadIdx.x], R_y[threadIdx.x], R_z[threadIdx.x], R_x[i], R_y[i], R_z[i]);
            int h_pos = (int)(current_distance2 / bucket_width);
            atomicAdd((unsigned long long *)&(local_hist[h_pos].d_cnt), (unsigned long long)1);
        }
		__syncthreads(); //to ensure all private copies are fully updated
    }

	//add the private histogram to the main histogram
    if(threadIdx.x == 0) {
        for (int i = 0; i < num_buckets; i++) {
            atomicAdd((unsigned long long *)&(d_histogram[i].d_cnt), (unsigned long long) local_hist[i].d_cnt);
        }
    }    
}

/* brute-force SDH solution in a single CPU thread */
int PDH_baseline(atom_array atom_array1) {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j, atom_array1);
			h_pos = (int) (dist / PDH_res); //distance / size of a bucket to get the correct bucket
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* set a checkpoint and show the (natural) running time in seconds */
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/* print the counts in all buckets of the histogram */
void output_histogram(){
	printf("\nCPU Histogram:\n");
	int i; 
	long long total_cnt = 0;
	for(i = 0; i < num_buckets; i++) {
		if(i % 5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void output_histogram_gpu(){
	printf("\nGPU Histogram:\n");
	int i; 
	long long total_cnt = 0;
	for(i = 0; i < num_buckets; i++) {
		if(i % 5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", gpu_histogram[i].d_cnt);
		total_cnt += gpu_histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void compare_histograms(bucket *cpu_hist, bucket *gpu_hist) {
    printf("\nDifference between CPU and GPU histograms:\n");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n");
}

int main(int argc, char **argv)
{
	int i; //used to initialie the random values

	//for CPU version
	atom_array atom_array1;
	
	if(argc != 4){
		fprintf(stderr, "\nError: Insufficient arguments, 3 arguments are needed\n\n");
        exit(EXIT_FAILURE);
	}

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	int BLOCK_DIM = atoi(argv[3]);

	if(BLOCK_DIM > 1024 || !(BLOCK_DIM % 32 == 0)|| BLOCK_DIM <= 0){
		fprintf(stderr, "\nError: Please enter a valid block size\n\n");
        exit(EXIT_FAILURE);
	}

	atom_array1.x_pos = (double *)malloc(sizeof(double) * PDH_acnt);
	atom_array1.y_pos = (double *)malloc(sizeof(double) * PDH_acnt);
	atom_array1.z_pos = (double *)malloc(sizeof(double) * PDH_acnt);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

	histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
	
	//allocate space for gpu histogram, which will be the same as the normal histogram (if the gpu computation is correct), this is on the cpu
	gpu_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++){
		atom_array1.x_pos[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_array1.y_pos[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_array1.z_pos[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline(atom_array1);

	//parallelized version
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//this will point to the struct which contains the coordinate arrays on the device
	atom_array * d_atom_array;

	//temporary host array so that it can be copied to device atom array
	atom_array * h_atom_array;

	h_atom_array = (atom_array *)malloc(sizeof(atom_array));
	double * x_pos_temp;
	double * y_pos_temp;
	double * z_pos_temp;
	
	//to allocate space for the struct
	cudaMalloc(&d_atom_array, sizeof(atom_array));

	//to allocate space for the arrays within the struct
	cudaMalloc(&x_pos_temp, sizeof(double) * PDH_acnt);
	cudaMalloc(&y_pos_temp, sizeof(double) * PDH_acnt);
	cudaMalloc(&z_pos_temp, sizeof(double) * PDH_acnt);

	//copy the contents of the coordinates to the device
	cudaMemcpy(x_pos_temp, atom_array1.x_pos, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(y_pos_temp, atom_array1.y_pos, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(z_pos_temp, atom_array1.z_pos, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
	
	//make the pointers within d_atom_array point to the corresponding dataset
	h_atom_array -> x_pos = x_pos_temp;
	h_atom_array -> y_pos = y_pos_temp;
	h_atom_array -> z_pos = z_pos_temp;

	cudaMemcpy(d_atom_array, h_atom_array, sizeof(atom_array), cudaMemcpyHostToDevice);

	bucket* d_histogram;

    cudaMalloc(&d_histogram, sizeof(bucket) * num_buckets);

	cudaMemcpy(d_histogram, gpu_histogram, sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);

	int block_count = (int)(PDH_acnt/BLOCK_DIM) + 1;

    size_t shared_mem_size = (3 * BLOCK_DIM * sizeof(double)) + (sizeof(bucket) * num_buckets);
	
	PDH_baseline_parallel<<<block_count, BLOCK_DIM, shared_mem_size>>>(PDH_acnt, PDH_res, d_atom_array, d_histogram, block_count, num_buckets);
	
	cudaMemcpy(gpu_histogram, d_histogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	/* print out the histogram */
	output_histogram();
	output_histogram_gpu();
	compare_histograms(histogram, gpu_histogram);
	
	/* check the total running time */ 
	report_running_time();
	printf(" ****** Total Running Time of Kernel: %0.5f ms ******\n", elapsedTime);

	free(h_atom_array);
	cudaFree(d_atom_array);
	cudaFree(x_pos_temp);
	cudaFree(y_pos_temp);
	cudaFree(z_pos_temp);
	cudaFree(d_histogram);
	free(histogram);
	free(gpu_histogram);

	return 0;
}
