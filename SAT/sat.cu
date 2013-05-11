 //Complete 3SAT solver using CUDA

// Parallel Processing course assignment

// Author : Gourab Saha
//Contact : 9051110501

// To compile : nvcc sat.cu
 
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
  
  int ind;
  
 using namespace std;
 
 
  void generate_random_cnf( int *,int);
  void  reset (int *status,int *new_status);
  void find_sequence(int *sequence,int temp);
  int find_status(int start,int end,int* sat_cnf2,int *sequence,int *status,int *new_status);
  int find_one(int *final_result,int size);
   void  print_details(int i,int j,int k,int index,int *sat_cnf2);
  
  
  
  
  
 __global__ void sat_Kernel1 (int *sat_cnf2_device,int *result_h1_device,int iteration)
 {
 
	 int sequence[5];
	 int temp,i,j,new1,val,start,end;
	 int status[25];
	 
	 unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	 result_h1_device[index]=1;
	 
	 temp=index;
	 
	 for(i=4;i>=0;i--)
	 {
		 sequence[i]=temp%3;
		 temp=temp/3;
	 }
	 
	 for(i=0;i<25;i++)
	 {
		 status[i]=2;
	 }
	 
	 start=iteration *5;
	 end=start+4;
	 for(i=start,j=0;i<end;i++,j++)
	 {
		 new1=sat_cnf2_device[i*6+sequence[j]*2];
		 val=sat_cnf2_device[i*6+sequence[j]*2+1];
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
	 
	 
	 __syncthreads();
 }
	 
  
  
   
 __global__ void sat_Kernel2 (int * sat_cnf2_device,int *result_h1_device,int *status_device,int *final_result_device)
 {
  
  
  
	 int sequence[5];
	 int temp,i,j,new1,val,start,end;
	 int status[25];
	 int status2[25];
	 int flag;
	 
	 unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	 temp= result_h1_device[index];
	 
	 
   
	 
	 for(i=4;i>=0;i--)
	 {
		 sequence[i]=temp%3;
		 temp=temp/3;
	 }
	 
	 for(i=0;i<25;i++)
	 {
		 status[i]=status_device[i];
		 status2[i]=2;
	 }
	 
	 
	 
	 start=15;
	 end=19;
	 for(i=start,j=0;i<end;i++,j++)
	 {
		 new1=sat_cnf2_device[i*6+sequence[j]*2];
		 val=sat_cnf2_device[i*6+sequence[j]*2+1];
		 status2[new1]=val;
		
	  }
	  
	  flag=1;
	  for(i=0;i<25;i++)
	  {
	   if((status[i]+status2[i])==1)
	   {
	      flag=0;
	      break;
	   }
	  
	  }
	  
	  final_result_device[index]=flag;
	 
	 __syncthreads();
  
     
  
 }
 
 
 
 
 
 
 
 int main(void)
 {
   
   
   int *sat_cnf,*sat_cnf2,*result_h1;
   int size,size2,no_of_iteration;
   int i,j,k,z1,*result_h1_device,*status_device;
   
   
   size_t array_size,block_size,num_blocks; 
   int *sat_cnf2_device,*final_result_device;
   int count[4];
   int status[25];
   int *final_result;
   int p1,p2,p3,flag,final_flag;
   int sequence[5],new_status_h[25],new_status_i[25],new_status_j[25],new_status_k[25];
 
  
   
   
   
   
   
   size=NO_OF_CLAUSE *NO_OF_VARIABLE;
   size2=NO_OF_CLAUSE * CLAUSE_SIZE * 2 ;
   sat_cnf=(int*) malloc(size * sizeof(int));
   sat_cnf2=(int*) malloc(size2 * sizeof(int));
   
   
 
   
   
   
   
   
   generate_random_cnf( sat_cnf,size);
   
  printf("\nRandom generated boolean expression with 25 variables and 20 clasues(with 3 literals) in Conjunctive normal form\n\n");
   
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
     
    array_size=no_of_iteration*243*sizeof(int);
    cudaMalloc((void **) &result_h1_device,array_size);
    
     
     
     for(i=0;i<no_of_iteration;i++)
     {
	 sat_Kernel1<<<1,block_size>>>(sat_cnf2_device,result_h1_device,i);
	 
	 array_size=243*sizeof(int);
	 cudaMemcpy(&result_h1[i*243],result_h1_device,array_size,cudaMemcpyDeviceToHost);
     }
	 
	 
	 
     for(i=0;i<no_of_iteration;i++)
   {
      count[i]=0;
      for(j=0;j<243;j++)
      { 	
	   if(result_h1[i*243+j]==1)
	   {
	     result_h1[i*243+count[i]]=j;
	     count[i]++;
	   }
	 
      }	
    }
    
    
    
 
    
    final_result=(int *)malloc(count[3]*sizeof(int));
    
    block_size=count[3];
    num_blocks=1;
    
     dim3 dimBlock(block_size);
     dim3 dimGrid(num_blocks);
    
    
    array_size=25*sizeof(int);
    cudaMalloc((void **) &status_device,array_size);
   
    
     array_size=count[3]*sizeof(int);
     cudaMalloc((void **) &final_result_device,array_size);
     
      array_size=243*sizeof(int);
      cudaMemcpy(result_h1_device,&result_h1[729],array_size,cudaMemcpyHostToDevice);
 
     
 
    final_flag=0;
    
     for(i=0;i<25;i++)
    {
      new_status_h[i]=2;
    }
    for(i=0;i<count[0];i++)
    {
       if(final_flag==1)
       {
	   i--;
	   break;
	  
       }
       reset(status,new_status_h);
       p1=result_h1[i];
       find_sequence(sequence,p1);
       flag=find_status(0,4,sat_cnf2,sequence,status,new_status_i);
       if(flag==0)
	continue;
       
      for(j=0;j<count[1];j++)
      {
	if(final_flag==1)
	{
	  j--;
	   break;
	}
	reset(status,new_status_i);
	p2=result_h1[243+j];
	find_sequence(sequence,p2);
	flag=find_status(5,9,sat_cnf2,sequence,status,new_status_j);
	if(flag==0)
	 continue;
	
       for(k=0;k<count[2];k++)
       {
	  reset(status,new_status_j);
	  p3=result_h1[2*243+k];
	 find_sequence(sequence,p3);
	 flag=find_status(10,14,sat_cnf2,sequence,status,new_status_k);
	 if(flag==0)
	  continue;
	 else
	 {
	  reset(status,new_status_k);
	  array_size=25*sizeof(int);
	  cudaMemcpy(status_device,status,array_size,cudaMemcpyHostToDevice);
	  
	  
	  sat_Kernel2<<<dimGrid,dimBlock>>>(sat_cnf2,result_h1_device,status_device,final_result_device);
	  
	  array_size=count[3]*sizeof(int);
	 cudaMemcpy(final_result,final_result_device,array_size,cudaMemcpyDeviceToHost);
	  printf("\n\n");
	 
	  final_flag=find_one(final_result,count[3]);
	  if(final_flag==1)
	   break;
	 }
       }
      }
    }
    
    
   
    
    
    printf("\n\n");
    
    if(final_flag==1)
    {
      printf(" \nThe CNF is satisfiable with the following assignment \n\n");
      print_details(i,j,k,ind,sat_cnf2);
    }
    
    else
    {
       printf(" \nThe CNF is not satisfiable !! \n\n");
    }
    
    
    
    
   
   /* 
      for(i=0;i<no_of_iteration;i++)
   {
     printf("\nno of 1s :%d\n",count[i]);
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
     
   */
   
   
   cudaFree(sat_cnf2_device);
   cudaFree(final_result_device);
   cudaFree(result_h1_device);
   cudaFree(status_device);
   
   free(sat_cnf);
   free(sat_cnf2);
   free(result_h1);
   free(final_result);
   
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
				 
				 
   
   
   
  void  reset (int *status ,int *new_status)
   {
   
    int i;
     for(i=0;i<25;i++)
	 {
		 status[i]=new_status[i];
	 }
	 
   }
		 
  void find_sequence(int *sequence,int temp)
  {
    int i;
    for(i=4;i>=0;i--)
	 {
		 sequence[i]=temp%3;
		 temp=temp/3;
	 }
  }
  
  
  
  
  int find_status(int start,int end,int* sat_cnf2,int *sequence,int *status,int *new_status)
  {
   int i,j,new1,val;
   
   for(i=0;i<25;i++)
    new_status[i]=status[i];
   for(i=start,j=0;i<end;i++,j++)
   {
	new1=sat_cnf2[i*6+sequence[j]*2];
	val=sat_cnf2[i*6+sequence[j]*2+1];
	if(new_status[new1]+val==1)
	{
	   return 0;
	 }
	 else
	 {
	     new_status[new1]=val;
	  }
		 
   }
   return 1;
   
  }
		 
  
  
  int find_one(int *final_result,int size)
  {
    int i,flag=0;
    for(i=0;i<size;i++)
    {
     if(final_result[i]==1)
     {
      flag=1;
      ind=i;
      break;
     }
    }
    return flag;
  }
 
 
 
 void  print_details(int i,int j,int k,int index,int *sat_cnf2)
 {
   int sequence[5];
    int temp,x,new1,y,val;
    int status[25];
    
    for(x=0;x<25;x++)
     status[x]=2;
  
   
    temp=i;
   
    for(x=4;x>=0;x--)
   {
		 sequence[x]=temp%3;
		 temp=temp/3;
    }
    
    for(x=0,y=0;x<4;x++,y++)
    {
	  new1=sat_cnf2[x*6+sequence[y]*2];
	   val=sat_cnf2[x*6+sequence[y]*2+1];
	   status[new1]=val;
		
     }
     
     
     
     temp=j;
   
    for(x=4;x>=0;x--)
   {
		 sequence[x]=temp%3;
		 temp=temp/3;
    }
    
    for(x=5,y=0;x<9;x++,y++)
    {
	  new1=sat_cnf2[x*6+sequence[y]*2];
	   val=sat_cnf2[x*6+sequence[y]*2+1];
	   status[new1]=val;
		
     }
  
  
   temp=k;
   
    for(x=4;x>=0;x--)
   {
		 sequence[x]=temp%3;
		 temp=temp/3;
    }
    
    for(x=10,y=0;x<14;x++,y++)
    {
	  new1=sat_cnf2[x*6+sequence[y]*2];
	   val=sat_cnf2[x*6+sequence[y]*2+1];
	   status[new1]=val;
		
     }
     
     
     
     
     temp=i;
   
    for(x=4;x>=0;x--)
   {
		 sequence[x]=temp%3;
		 temp=temp/3;
    }
    
    for(x=15,y=0;x<19;x++,y++)
    {
	  new1=sat_cnf2[x*6+sequence[y]*2];
	   val=sat_cnf2[x*6+sequence[y]*2+1];
	   status[new1]=val;
		
     }
     
     
       
    for(x=0;x<25;x++)
    {
     if(status[x]==2)
      printf("\nx%d : Dont Care(0 or 1)",x);
     else
     printf("\nx%d : %d",x,status[x]);
    }
    
    printf("\n\n\n");
  
 }
   
   
   
   
   
   
   
   
