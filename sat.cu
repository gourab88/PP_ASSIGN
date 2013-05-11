#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include <stdlib.h>
#include <set>
#include <iostream>
#include <map>
#include <time.h>
 
 
 
 
 
 
#define NO_OF_VARIABLE 25
#define NO_OF_CLAUSE 20
#define CLAUSE_SIZE 3
#define MAX_THREAD_PER_BLOCK 512
 
 
 
using namespace std;

 void generate_random_cnf( int *,int);
 
 
 
 
 
__global__ void sat_Kernel1 (int *sat_cnf2_device,int *result_h1_device,int iteration)
{

	int sequence[5];
	int temp,i,new1,val,start,end;
	int status[25];
	
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	result_h1_device[index]=1;
	
	temp=index;
	
	for(i=4;i>=0;i--)
	{
		sequence[i]=temp%5;
		temp=temp/5;
	}
	
	for(i=0;i<25;i++)
	{
		status[i]=2;
	}
	
	start=iteration *5;
	end=start+5;
	for(i=start;i<end;i++)
	{
		new1=sat_cnf2_device[i*6+sequence[i]*2];
		val=sat_cnf2_device[i*6+sequence[i]*2+1];
		if(status[new1]+val==1)
		{
			result_h1_device[index]=0;
			break;
		}
		else
		{
			status[new1]=val;
		}
		
	}
	
}
	
	





int main(void)
{
  
  
  int *sat_cnf,*sat_cnf2,*result_h1;
  int size,size2,no_of_iteration;
  int i,j,z1,*result_h1_device;
  
  
  size_t array_size,block_size,num_blocks; 
  int *sat_cnf2_device;
  
  
  
  
  
  size=NO_OF_CLAUSE *NO_OF_VARIABLE;
  size2=NO_OF_CLAUSE * CLAUSE_SIZE * 2 ;
  sat_cnf=(int*) malloc(size * sizeof(int));
  sat_cnf2=(int*) malloc(size2 * sizeof(int));
  
  

  
  
  
  
  
  generate_random_cnf( sat_cnf,size);
  
 
  
   for(i=0;i<NO_OF_CLAUSE;i++)
  {
   
  	printf("(");
  	z1=0;
  	for(j=0;j<NO_OF_VARIABLE;j++)
  	{
  		if(sat_cnf[i*NO_OF_VARIABLE+j]==1)
  		{
  				 printf("x%d",j);
  				sat_cnf2[i*CLAUSE_SIZE * 2+2*z1]=j;
  				sat_cnf2[i*CLAUSE_SIZE * 2+2*z1+1]=1;
  				
  				z1++;
  			   
  		}
  		if(sat_cnf[i*NO_OF_VARIABLE+j]==0)
  		{
  				 printf("~x%d",j);
  				sat_cnf2[i*CLAUSE_SIZE * 2+2*z1]=j;
  				sat_cnf2[i*CLAUSE_SIZE * 2+2*z1+1]=0;
  				
  				z1++;
  		}
  	 	if(z1<CLAUSE_SIZE && sat_cnf[i*NO_OF_VARIABLE+j]!=2)
  			 printf("+");
  		
  		
  	}
   
 	printf(")");
  	if(i<NO_OF_CLAUSE-1)
  		 printf(" * ");
  }
  		
  
  
  
  
  array_size=NO_OF_CLAUSE * CLAUSE_SIZE * 2 * sizeof(int) ;
  cudaMalloc((void **) &sat_cnf2_device,array_size);
  cudaMemcpy(sat_cnf2_device,sat_cnf2,array_size,cudaMemcpyHostToDevice);
  
  
   block_size=243;
   num_blocks=1;
   no_of_iteration=4;
   
    size= no_of_iteration*243;
    result_h1=(int*) malloc(size * sizeof(int));
    
    array_size=243*sizeof(int);
    cudaMalloc((void **) &result_h1_device,array_size);
   
    dim3 dimBlock(block_size);
    dim3 dimGrid(num_blocks);
    
    
    for(i=0;i<no_of_iteration;i++)
    {
    	sat_Kernel1<<<dimGrid,dimBlock>>>(sat_cnf2_device,result_h1_device,i);
    	
    	array_size=243*sizeof(int);
    	cudaMemcpy(&result_h1[i*243],result_h1_device,array_size,cudaMemcpyDeviceToHost);
    }
    	
    	
    	
    for(i=0;i<no_of_iteration;i++)
  {
   		for(j=0;j<243;j++)
  		{ 	
    		printf("%d ",result_h1[i*243+j]);
    	
    	}	
    printf("\n");
   }
   
         printf("\n");
     
     for(i=0;i<NO_OF_CLAUSE;i++)
  {
   		
   		for(j=0;j<6;j=j+2)
  	{
  		printf("x%d",sat_cnf2[i*6+j]);
  		if(sat_cnf2[i*6+j+1]==0)
  			printf("'");
  		printf("+");
  		}
  		printf("\n");
  }
    
    
    return 1;
  
  
  }
  
  
  
  
void generate_random_cnf( int *sat_cnf,int size)
  {
  		int i,j,flag,k;
  		int a[CLAUSE_SIZE];
  		
  		
  		for(i=0;i<size;i++)
  		{
  			sat_cnf[i]=2;
  			
		}
		
		for(i=0;i<NO_OF_CLAUSE;i++)
  		{
  			for(j=0;j<CLAUSE_SIZE;j++)
  			{
  				flag=1;
  				while(flag)
  				{
  						a[j]=rand()%NO_OF_VARIABLE;
  						flag=0;
  					    for(k=j-1;k>=0;k--)
  					    {
  					    	if(a[j]==a[k])
  					    	{
  					    		flag=1;
  					    		break;
  					    	}
  					    }
  			   }
  					
  				if(rand()%2==0)
  					sat_cnf[i*NO_OF_VARIABLE+a[j]]=0;
  				else
  					sat_cnf[i*NO_OF_VARIABLE+a[j]]=1;
  			}
  		}
  			
  }
  				
  				
		
		
		
		
 		

  
  
  
  
  
  
  
  
