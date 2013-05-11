// Stimulation of information diffusion in social network with 
//respect to time in social network using CUDA

// Parallel Processing course assignment

// Author : Gourab Saha
//Contact : 9051110501

// To compile : nvcc prog5.0.cu 



#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include <stdlib.h>
#include <set>
#include<curand.h>
#include<curand_kernel.h>
#include <iostream>
#include <map>
#include <time.h>
 



#define NO_OF_USER 100
#define MAX_NEIGHBOUR 4
#define MIN_NEIGHBOUR 10
#define SIZE_OF_LOCAL_LIST 10
#define SIZE_OF_GLOBAL_LIST 500
#define MAX_TIMESTAMP 2880
#define LAMDA_ARRIVAL_GLOBAL 0.08
#define LAMDA_ARRIVAL_USER  0.1
#define TIMEUNIT_BEFORE_ZERO 1000
#define INITIAL_WEIGHT_GL 100
#define INITIAL_WEIGHT_LL 100
#define GLOBAL_DECAY_RATE 0.05
#define LOCAL_DECAY_RATE 0.05
#define SAMPLING_INTERVAL 6
#define WEIGHT_THRESOLD 0.01
 

using namespace std;
int no_of_topics;
int no_of_iteration;
int global_zero_index;

void make_graph(int *v_graph , int *e_graph);
void make_global_list(float * global_list);
void initialize_graph(float *global_list,float *local_list,int **topic_book_keeping);
void generate_initial_action_time(float **user_action);
 






/*
      **********************************************
      **                                          **
      **            Kernel Function               **
      **                                          **
      ********************************************** 
 */  
   
 
 

   
__global__ void  Take_Action_Kernel(int *v_graph_device,int *e_graph_device,float *local_list_device,
                                    float * global_list_device,short *active_now_device,short *active_element_ids_device,
                                    short *count_device,int *no_of_topics_device,float *current_time_device,int *result_device
                                    ,curandState * state, unsigned long seed)
{
      clock_t start_parallel_device= clock();
     
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
 
     
  
    
    float weights[500];
    float roulette_wheel[500];
    int current_index,count=0,i,j,n,user,topic;
    float weight,sum=0.0f,x=0.0f;
    int node_index,neighbours,last_change,N;
    
    node_index=active_element_ids_device[index];
    
 
    neighbours=v_graph_device[node_index+1]-v_graph_device[node_index];
    
    for(i=0;i<*no_of_topics_device;i++)
        weights[i]=0.0f;
    
    for(i=0;i<neighbours;i++)
    {
        current_index=v_graph_device[node_index]+i;
        j=(*no_of_topics_device)*e_graph_device[current_index];
        N=*no_of_topics_device+j;
        for(count=j;count<N;count++)
        {
            if(local_list_device[count] < -1000.0f)
                continue;
            weight=INITIAL_WEIGHT_LL*(float)expf(-1*LOCAL_DECAY_RATE * (*current_time_device-local_list_device[count]));
            if(weight<WEIGHT_THRESOLD)
                local_list_device[count]=-9999.0f;
            else
            {
               weights[count-j]=weights[count-j]+weight;
               sum=sum+weight;
            }
               
                
        }
    }
    
    for(count=0;count<*no_of_topics_device;count++)
    {
         if(global_list_device[count]>*current_time_device)
  			break;
 
        if(global_list_device[count] < -1000.0f)
                continue;
        weight=INITIAL_WEIGHT_GL*(float)expf(-1*GLOBAL_DECAY_RATE * (*current_time_device-global_list_device[count]));
        if(weight<WEIGHT_THRESOLD)
            global_list_device[count]=-9999.0f;
        else
            {
               weights[count]=weights[count]+weight;
               sum=sum+weight;
            }
    }
    
    roulette_wheel[0]=0.0f;
    x=0.0f;
     for(count=0;count<*no_of_topics_device;count++)
     {
         weights[count]=weights[count]/sum;
        roulette_wheel[count+1]=x+ weights[count];
        x=roulette_wheel[count+1]; 
     }
     
     
    
    curandState localState = state[threadIdx.x];
    curand_init ( seed,threadIdx.x, 0, &localState);
   
    x= curand_uniform( &localState );
   state[threadIdx.x] = localState; 
    
    last_change=0;
     i=0;
     
    while( roulette_wheel[i]<x)
    {
        if(i>0 && roulette_wheel[i]!=roulette_wheel[i-1])
        	last_change=i;
        i++;
       }
     i=last_change;
    
    __syncthreads();
    
    local_list_device[((*no_of_topics_device)*node_index)+last_change]=*current_time_device;
    
    __syncthreads();
    
 
   //Update with new weights
   //bookkeeping of no of copies of topics
   
   clock_t end_parallel_device= clock();
   
   if(index==0)
    {
         n=NO_OF_USER*(*no_of_topics_device);
       
        
         for(count=0;count<*no_of_topics_device;count++)
         {
            result_device[count]=0;
         }
          
         
         
         for(count=0;count<n;count++)
         {
            user=count/(*no_of_topics_device);
            topic=count%(*no_of_topics_device);
            if(active_now_device[user]==1)
            {
                if(local_list_device[count] > -1000.0f)
                result_device[topic]++;
            }
            else
            {
                if(local_list_device[count] > -1000.0f)
                    
                {
                
                     weight=INITIAL_WEIGHT_LL*exp(-1*LOCAL_DECAY_RATE *  (*current_time_device-local_list_device[count]));
                    if(weight<WEIGHT_THRESOLD)
                     local_list_device[count]=-9999.0f;
                     else
                     {
                          result_device[topic]++;
                    }
                }
            }
            
         }
    }
    else
    	return;

 
 
             
}
    
    




/*
      **********************************************
      **                                          **
      **       Serial execution Function          **
      **                                          **
      ********************************************** 
 */  
   








void Take_Action_serial(int *v_graph,int *e_graph ,float*local_list,float*global_list,int no_of_topics,float current_time,int *result,int user,short * active_now)
{

	
	//printf("user :%d\n\n",user);
	float weights[800];
    float roulette_wheel[800];
    int current_index,count=0,i,j;
    float weight,sum=0.0f,x=0.0f;
    int neighbours,last_change,N;
    
    if(active_now[user]==0)
    	return;
    
    
    neighbours=v_graph[user+1]-v_graph[user];
    for(i=0;i<no_of_topics;i++)
        weights[i]=0.0f;
        
     for(i=0;i<neighbours;i++)
    {
        current_index=v_graph[user]+i;
        j=(no_of_topics)*e_graph[current_index];
        N=no_of_topics+j;
        for(count=j;count<N;count++)
        {
            if(local_list[count] < -1000.0f)
                continue;
            weight=INITIAL_WEIGHT_LL*(float)exp(-1*LOCAL_DECAY_RATE * (current_time-local_list[count]));
            if(weight<WEIGHT_THRESOLD)
                local_list[count]=-9999.0f;
            else
            {
               weights[count-j]=weights[count-j]+weight;
               sum=sum+weight;
            }
               
                
        }
    }


	 for(count=0;count<no_of_topics;count++)
    {
         if(global_list[count]>current_time)
  			break;
        if(global_list[count] < -1000.0f)
                continue;
        weight=INITIAL_WEIGHT_GL*(float)exp(-1*GLOBAL_DECAY_RATE * (current_time-global_list[count]));
        if(weight>=WEIGHT_THRESOLD)
            {
               weights[count]=weights[count]+weight;
               sum=sum+weight;
            }
    }
	

	roulette_wheel[0]=0.0f;
    x=0.0f;
     for(count=0;count<no_of_topics;count++)
     {
         weights[count]=weights[count]/sum;
        roulette_wheel[count+1]=x+ weights[count];
        x=roulette_wheel[count+1]; 
     }
     
     x=(float)rand()/(float)RAND_MAX;
     //printf("\n\nx=%f\n\n",x);
     last_change=0;
     i=0;
     
    while( roulette_wheel[i]<x)
    {
        if(i>0 && roulette_wheel[i]!=roulette_wheel[i-1])
        	last_change=i;
        i++;
       }
     i=last_change;
     
    local_list[((no_of_topics)*user)+last_change]=current_time;
    
 }
    





/*
      **********************************************
      **                                          **
      **             Main function                **
      **                                          **
      ********************************************** 
 */  
   







int main(void)
{
    
 
    
    clock_t start_main = clock();
    
    int *v_graph ,*e_graph,*result;
    float *local_list,*global_list;
    short  *active_now,*active_element_ids,count;
    float current_inteval_start,current_interval_end,current_time;
    int i,j,a,topic;
    unsigned long long int n;
    int **topic_book_keeping;
    
    float **user_action,weight;
    int *user_current_state;
    
    curandState* devStates;
    
    
    size_t array_size,block_size,num_blocks; 
    float *local_list_device,*global_list_device;
    int *v_graph_device ,*e_graph_device,*result_device;
    short  *active_element_ids_device, *count_device,* active_now_device;
    float *current_time_device;
    int *no_of_topics_device,*zero_array;
   // uint kernelTime;
    
    
    double common_time;
    double parallel_time=0.0f;
    double serial_time;
    double total_parallel;
    double total_serial;
    double speed_up;
    double speed_up2;
    
  
    
    
    v_graph=(int*)malloc((NO_OF_USER+1)*sizeof(int));
    e_graph=(int*)malloc(NO_OF_USER*10*sizeof(int));
    
    global_list=(float*)malloc(SIZE_OF_GLOBAL_LIST*sizeof(float));
    active_now=(short *)malloc(NO_OF_USER*sizeof(short));
    


    
    
    
    
    
   make_graph(v_graph,e_graph);
   make_global_list(global_list);
   
   
   no_of_iteration=MAX_TIMESTAMP/SAMPLING_INTERVAL;
   
    user_action=(float**)malloc(NO_OF_USER*sizeof(float*));
    for(j=0;j<NO_OF_USER;j++)
		 user_action[j]=(float*)malloc((no_of_iteration+1)*sizeof(float)); 
	
    
    
    user_current_state=(int*)calloc(NO_OF_USER,sizeof(int)); 
   
   result=(int *)malloc(no_of_topics*sizeof(int));
   for(j=0;j<no_of_topics;j++)
		result[j]=0; 
	
	zero_array= (int *)calloc(no_of_topics,sizeof(int)); 
		
		
   topic_book_keeping=(int **)calloc(no_of_topics,sizeof(int*));
   for(i=0;i<no_of_topics;i++)
        topic_book_keeping[i]=(int *)calloc((no_of_iteration+1),sizeof(int));
   
   local_list=(float*)malloc(NO_OF_USER*no_of_topics*sizeof(float));
   
   initialize_graph(global_list,local_list,topic_book_keeping);
  
  
   generate_initial_action_time(user_action);
 
 

    
    
    array_size=(NO_OF_USER+1)*sizeof(int);
    cudaMalloc((void **) &v_graph_device,array_size);
    cudaMemcpy(v_graph_device,v_graph,array_size, cudaMemcpyHostToDevice);
    
    array_size=NO_OF_USER*10*sizeof(int);
    cudaMalloc((void **) &e_graph_device,array_size);
    cudaMemcpy(e_graph_device,e_graph,array_size, cudaMemcpyHostToDevice);
    
    array_size=NO_OF_USER*no_of_topics*sizeof(float);
    cudaMalloc((void **) &local_list_device,array_size);
    cudaMemcpy(local_list_device,local_list,array_size, cudaMemcpyHostToDevice);
    
    array_size=SIZE_OF_GLOBAL_LIST*sizeof(float);
    cudaMalloc((void **) &global_list_device,array_size);
    cudaMemcpy(global_list_device,global_list,array_size, cudaMemcpyHostToDevice);
    
       
    array_size=sizeof(int);
    cudaMalloc((void **) &no_of_topics_device,array_size);
    cudaMemcpy(no_of_topics_device,&no_of_topics,array_size,cudaMemcpyHostToDevice);
    
   
    array_size=NO_OF_USER*sizeof(short int);
    cudaMalloc((void **) &active_now_device,array_size);
    
    array_size=sizeof(float);
    cudaMalloc((void **) &current_time_device,array_size);
    
    array_size=no_of_topics*sizeof(int);
    cudaMalloc((void **) &result_device,array_size);
    
    
    
     printf("\n\nInitial status :"); 
           for(j=0;j<no_of_topics;j++)
           		printf("%d,",topic_book_keeping[j][0]); 
   
   
   
   
    clock_t end_main= clock();
    
    
    
    
    
    
    
    
    /*
      **********************************************
      **                                          **
      **        Parallel execution starts         **
      **                                          **
      ********************************************** 
  */  
   
   
   
   
 
    for(i=0;i<no_of_iteration;i++)
    {
        count=0;
        current_inteval_start=SAMPLING_INTERVAL*i;
        current_interval_end=SAMPLING_INTERVAL*(i+1);
        current_time=current_inteval_start+(SAMPLING_INTERVAL/2);
        
        
         printf("\n\n Current time interval : %lf-%lf",current_inteval_start,current_interval_end);
    
        for(j=0;j<NO_OF_USER;j++)
        {
            if(user_action[j][user_current_state[j]]>=current_inteval_start && user_action[j][user_current_state[j]]<current_interval_end)
            {
                active_now[j]=1;
                count++;
               user_current_state[j]++;
                
            }
            else
            {
                while(user_action[j][user_current_state[j]]<current_inteval_start)
                	 user_current_state[j]++;
                active_now[j]=0;
                
              }
        }
        
        active_element_ids=(short*)malloc(count*sizeof(short));
      
        a=0;
        for(j=0;j<NO_OF_USER;j++)
        {
            if(active_now[j]==1)
            {
                  active_element_ids[a]=j;
                  a++;
            }
        }
        
        printf("\n No of active users :%d",count );
       
        array_size=count*sizeof(short);
        cudaMemcpy(active_now_device,active_now,array_size, cudaMemcpyHostToDevice);
        
        cudaMalloc((void **) &active_element_ids_device,array_size);
        cudaMemcpy(active_element_ids_device,active_element_ids,array_size,cudaMemcpyHostToDevice);
        
        array_size=sizeof(short);
        cudaMalloc((void **) &count_device,array_size);
        cudaMemcpy(count_device,&count,array_size, cudaMemcpyHostToDevice);
        
        
         array_size=sizeof(float);

        cudaMemcpy(current_time_device,&current_time,array_size,cudaMemcpyHostToDevice);
        
        
        cudaMalloc ( &devStates, count*sizeof( curandState ) );
        
    
      array_size=no_of_topics*sizeof(int);
      cudaMemcpy(result_device,zero_array,array_size,cudaMemcpyHostToDevice);
     
     
       block_size=count;
       num_blocks=1;
       
       dim3 dimBlock(block_size);
       dim3 dimGrid(num_blocks);
       
     clock_t start_parallel= clock();
       
      //  cutCreateTimer(&kernelTime);
 		//cutResetTimer(kernelTime);
       
     Take_Action_Kernel<<<dimGrid,dimBlock>>>(v_graph_device,e_graph_device,local_list_device,global_list_device,
                                               active_now_device,active_element_ids_device,count_device,no_of_topics_device,
                                                current_time_device,result_device,devStates,unsigned(time(NULL)));
                   
                   
       cudaThreadSynchronize();                                         
      // cutStopTimer(kernelTime);                      
       clock_t end_parallel = clock();
       
       parallel_time=(double)(end_parallel - start_parallel) / CLOCKS_PER_SEC;
       
        array_size=no_of_topics*sizeof(int);
        cudaMemcpy(result, result_device,array_size,cudaMemcpyDeviceToHost);
        
        
 
         
  
            
        for(j=0;j<no_of_topics;j++)
            topic_book_keeping[j][i+1]=result[j];
         
    printf("\n\nfor iteration %d :",i+1); 
           for(j=0;j<no_of_topics;j++)
  				printf("%d,",result[j]); 
      
        
         free(active_element_ids);
         cudaFree(active_element_ids_device);
         
    
 }
     
   
  
  
  clock_t start_serial = clock(); 
  
  
  
  
/*
      **********************************************
      **                                          **
      **         Serial execution starts          **
      **                                          **
      ********************************************** 
 */  
   
 
 
 
 
 initialize_graph(global_list,local_list,topic_book_keeping);
 make_global_list(global_list);
 generate_initial_action_time(user_action);
 
 
   printf("\n\nInitial status :"); 
           for(j=0;j<no_of_topics;j++)
           		printf("%d,",topic_book_keeping[j][0]); 
      
 
  
 
    for(i=0;i<no_of_iteration;i++)
    {
    	
  		count=0;
        current_inteval_start=SAMPLING_INTERVAL*i;
        current_interval_end=SAMPLING_INTERVAL*(i+1);
        current_time=current_inteval_start+(SAMPLING_INTERVAL/2);
        
        
         printf("\n\n Current time interval : %lf-%lf",current_inteval_start,current_interval_end);
    
        for(j=0;j<NO_OF_USER;j++)
        {
            if(user_action[j][user_current_state[j]]>=current_inteval_start && user_action[j][user_current_state[j]]<current_interval_end)
            {
                active_now[j]=1;
                count++;
               user_current_state[j]++;
                
            }
            else
            {
                while(user_action[j][user_current_state[j]]<current_inteval_start)
                	 user_current_state[j]++;
                active_now[j]=0;
                
              }
        }
        
       
    	 count=0;    
          for(j=0;j<NO_OF_USER;j++)
        {
            if(active_now[j]==1)
            {
                count++;
            }
        }
        
        printf("\n No of active users :%d",count );
        
        
     
   
  		
  		for(j=0;j<NO_OF_USER;j++)
  		{
  			 	Take_Action_serial(v_graph,e_graph ,local_list,global_list,no_of_topics,current_time,result,j,active_now);
 
  		}
  		
  
  	
  		n=NO_OF_USER*no_of_topics;
       
        
         for(count=0;count<no_of_topics;count++)
         {
            result[count]=0;
         }
          
  
   
          for(count=0;count<n;count++)
         {
           // user=count/(no_of_topics);
            topic=count%(no_of_topics);
            // printf("topics=%d\n\n",topic);
            if(local_list[count] > -1000.0f)
                    
                {
                
                     weight=INITIAL_WEIGHT_LL*exp(-1*LOCAL_DECAY_RATE *  (current_time-local_list[count]));
                    if(weight<WEIGHT_THRESOLD)
                     local_list[count]=-9999.0f;
                     else
                     {
                          result[topic]++;
                    }
                }
            
            
         }
         
       //  printf("I m here\n\n");
         
         
     for(count=0;count<no_of_topics;count++)
    {
         if(global_list[count]>current_time)
  			break;
          
          if(global_list[count] < -1000.0f)
                continue;
        weight=INITIAL_WEIGHT_GL*(float)exp(-1*GLOBAL_DECAY_RATE * (current_time-global_list[count]));
        if(weight<WEIGHT_THRESOLD)
          global_list[count]=-9999.0f;
    }
         
         
         
          for(j=0;j<no_of_topics;j++)
            topic_book_keeping[j][i+1]=result[j];
         
    	printf("\n\nfor iteration %d :",i+1); 
        for(j=0;j<no_of_topics;j++)
  			printf("%d,",result[j]); 
  			
 
  	
  	}
  
  
  
  
 clock_t end_serial = clock(); 
  
  
  

  
  common_time=(double)(end_main - start_main) / CLOCKS_PER_SEC;
  //parallel_time=(double)(end_parallel - start_parallel) / CLOCKS_PER_SEC;
  serial_time=(double)(end_serial - start_serial) / CLOCKS_PER_SEC;
  total_parallel=common_time+parallel_time;
  total_serial=common_time+serial_time;
  speed_up=total_serial/total_parallel;
  speed_up2=serial_time/parallel_time;
  
  
    
 /*
  printf("\n\nCommon time Elapsed: %f seconds\n",  common_time);  
  printf("\n\nParallel time Elapsed: %f seconds\n", parallel_time);
  printf("\n\nSerial time Elapsed: %f seconds\n",  serial_time);
  printf("\n\nTotal time Elapsed(Palallel execution): %f seconds\n",total_parallel);
  printf("\n\nTotal time Elapsed(serial execution): %f seconds\n",total_serial);
  printf("\n\nSpeed up :%f\n\n",speed_up);
  printf("\n\nSpeed up2 :%f\n\n",speed_up2);
  // printf ("Time for the kernel: %f ms\n", cutGetTimerValue(kernelTime));
  
  */
 
   free(v_graph);
   free(e_graph);
   free(local_list);
   free(global_list);
   free(active_now);
   free(result);
   free(zero_array);
   
   
   cudaFree(v_graph_device);
   cudaFree(e_graph_device);
   cudaFree(local_list_device);
   cudaFree(global_list_device);
   cudaFree(no_of_topics_device);
   cudaFree(current_time_device);
   cudaFree(active_now_device);
   cudaFree(result_device);
 
  
   
    
}





/*
      **********************************************
      **                                          **
      **             Graph creation               **
      **                                          **
      ********************************************** 
 */  
   





void make_graph(int *v_graph , int *e_graph)
{
 
 map<int,set<int> > graph;
 map<int,set<int> >::iterator map_itr;
 set<int>::iterator set_itr;
 set<int> nullset;
 set<int> hash;
 set<int> :: iterator itr;
 int no,i,neighbour,count,max,min;
 float r;
 int *neighbours;
 int edge_index;
 
 
 
 neighbours=(int*)calloc(NO_OF_USER,sizeof(int));
 

 
 for(i=0;i<NO_OF_USER;i++)
 	graph.insert(std::pair<int,set<int> >(i,nullset));

 for(i=0;i<NO_OF_USER;i++)
 {
    max=MAX_NEIGHBOUR-neighbours[i];
    min=MIN_NEIGHBOUR-neighbours[i];
    if(min<0)
     min=0;
     if(max>0)
     {
       r=(float)rand()/(float)(RAND_MAX);
       no=(int)(((float)(max-min)*r)+min);
       count=0;
       while(count<no)
        {
	    	srand(rand());
            neighbour=(int)(rand() % NO_OF_USER) ;
	        set_itr =graph[i].find(neighbour);
            if(neighbour==i || set_itr!=graph[i].end())
                continue;
	      
	     graph[i].insert(neighbour);
	     graph[neighbour].insert(i);
	     neighbours[i]++;
	     neighbours[neighbour]++;
	     count++;
		}
     }
 }
 
 i=0;
 edge_index=0;
 
 for (map_itr=graph.begin(); map_itr!=graph.end(); ++map_itr)
 {
   v_graph[i]=edge_index;
   
   for (set_itr=graph[i].begin(); set_itr!=graph[i].end(); ++set_itr)
   {
   	  e_graph[edge_index]=*set_itr;
   	  edge_index++;
   }
   map_itr->second.clear();
   i++;
 }
       
 graph.clear();
 
}






/*
      **********************************************
      **                                          **
      **              Global list                 **
      **                                          **
      ********************************************** 
 */  
   




void make_global_list(float *global_list)
{
    float t=0.0f,r=0.0f;
    int i=0,n,max;
    n=SIZE_OF_GLOBAL_LIST;
    max=TIMEUNIT_BEFORE_ZERO+MAX_TIMESTAMP;
      
      while (i<n && t<max)
      {
       //srand(rand());
       r=(float)rand()/(float)(RAND_MAX);
       t=t-(log(r)/LAMDA_ARRIVAL_GLOBAL);
        global_list[i]=t-TIMEUNIT_BEFORE_ZERO;
        if(global_list[i-1]<0 && global_list[i]>=0)
            global_zero_index=i;
        i++;
       }
       no_of_topics=i;
       
}





/*
      **********************************************
      **                                          **
      **         Initialization of graph          **
      **                                          **
      ********************************************** 
 */  
   
   
   




void initialize_graph(float *global_list,float *local_list,int **topic_book_keeping)
{
    
    float *weights,*roulette_wheel;
    int no,i,j;
    float sum=0.0f,x=0.0f;
    
    no=NO_OF_USER*no_of_topics;
   
    for(i=0;i<no;i++)
        local_list[i]=-9999.0f;
    
    weights=(float *)malloc(global_zero_index*sizeof(float));
    roulette_wheel=(float *)malloc((global_zero_index+1)*sizeof(float));
    roulette_wheel[0]=0;
    
    for(i=0;i<global_zero_index;i++)
    {
        weights[i]=INITIAL_WEIGHT_GL*exp(GLOBAL_DECAY_RATE * global_list[i]);
        sum=sum+ weights[i];
    }
     
     for(i=0;i<global_zero_index;i++)
    {
        weights[i]=weights[i]/sum;
        roulette_wheel[i+1]=x+ weights[i];
        x=roulette_wheel[i+1];
    }
    
    
   for(j=0;j<NO_OF_USER;j++)
   {
    i=0;
    srand(rand());
    x=(float)rand()/(float)(RAND_MAX);
    while( roulette_wheel[i]<x)
        i++;
    i--;
        
    local_list[(j*no_of_topics)+i]=0.0f;
    topic_book_keeping[i][0]++;
    
    
   }
   
      
   free(weights);
   free(roulette_wheel);
    
    
}




/*
      **********************************************
      **                                          **
      **    Genartion of initial action time      **
      **                                          **
      ********************************************** 
 */  
   




void generate_initial_action_time(float **user_action)
{
    float t=0.0f,r=0.0f;
    int i=0,j;
      
    for(i=0;i<NO_OF_USER;i++)
    {
       t=0.0f;
       r=0.0f;
       
       for(j=0;j<no_of_iteration+1;j++)
       {
        srand(rand());
        r=(float)rand()/(float)(RAND_MAX);
        t=-(log(r)/LAMDA_ARRIVAL_USER);
        if(j==0)
        	user_action[i][j]=t;
        else
        	user_action[i][j]=user_action[i][j-1]+t;
    	}
    }
     
}





 

