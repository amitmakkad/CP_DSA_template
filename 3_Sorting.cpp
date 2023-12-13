// Contents:-
// masters theorm
// proof of algorithm correctness
// insertion sort
// bubble sort 
// selection sort 
// merge sort 
// quick sort 
// heap sort 
// comparission based sorting 
// bucket sort 
// radix sort 
// counting sort 
// huffman coding
// median of median
// p-np 

//Masters theorm
T(n)=a*T(n/b)+f(n)
a>=1
b>1
f(n) asymptotic +ve function

-> f(n)=O(n^(logba-E))
    
    T(n)=O(n^(logba))

->f(n)=O(n^(logba))

    T(n)=O(n^(logba)*logn)

->f(n)=sigma(n^(logba+E))  and a*f(n/b)<=c*f(n) for some c<1

    T(n)=O(f(n))


T(n)=2T(n/2)+nlogn
no master theorm,as no epsilon found

• T(n) = 2nT(n/2) + n^n
• Does not apply (a is not constant)

• T(n) = 0.5T(n/2) + 1/n
• Does not apply (a < 1)

• T(n) = 16T(n/4) + n!
• T (n) = Θ(n!) (Case 3)


• T(n) = 64T(n/8) - n2log n
• Does not apply (f(n) is not positive)

• T(n) = T(n/2) + n(2 - cos n) 
regularity condn fail

T(n)=T(sqrt(n))+O(1)
subs n=pow(2,m)
O(loglogn)



// Algorithms 

The algorithm is correct if for any legal input it terminates and produces the desired output.

Proving that an algorithm is totally correct:
1. Proving that it will terminate
2. Proving that the list of actions applied to the precondition imply the
postcondition

• Example: Binary Search
• Input data: a:array of integer; x:integer;
• Output data: found:boolean;
• Precondition: a is sorted in ascending order
• Postcondition: found is true if x is in a, and found is false otherwise

Loop Invariants

• Invariants - assertions that are valid any time they are reached (many
times during the execution of an algorithm, e.g., in loops)
• We must show three things about loop invariants:


• Initialization - it is true prior to the first iteration

• Maintenance - if it is true before an iteration, it remains true before the next
iteration

• Termination - when loop terminates the invariant gives a useful property to show
the correctness of the algorithm

// Example: Loop invariant for Sum of n numbers

Algorithm
Sum_of_N_numbers

Initialization: Before the first iteration: k=0, S=0. The first 0
numbers have sum zero (there are no numbers) => hypothesis
true before entering the loop

Maintenance:
We assume that it is true at beginning of step k: “S is the sum
of the first k numbers”.
We have to prove that after executing step k, at the beginning
of step k+1: “S is the sum of the first k+1 numbers”
We calculate the value of S at the end of this step
K:=k+1, s:=s+a[k+1] => s is the sum of the first k+1 numbers

Termination: The loop terminates when k=n=> s= sum of first
k=n numbers => postcondition of algorithm, DONE


// merge sort

Precondition:
Array A has at least 1 element between indexes p and r (p<=r)

Postcondition:
The elements between indexes p and r are sorted

• Number of elements to be sorted: n=r-p+1

• Base Case: n = 1
• A contains a single element (which is trivially “sorted”)

• Inductive Hypothesis:
• Assume that MergeSort correctly sorts n=1, 2, ..., k elements

• Inductive Step:
• Show that MergeSort correctly sorts n = k + 1 elements.
• First recursive call n1=q-p+1=(k+1)/2<=k => subarray A[p.. q] is sorted
• Second recursive call n2=r-q=(k+1)/2<=k => subarray A[q+1 .. r] is sorted
• A, p q, r fulfill now the precondition of Merge

• The postcondition of Merge guarantees that the array A[p .. r] is sorted => postcondition of MergeSort


// merge
Precondition: A is an array and p, q, and r are indices into the array such that p <= q < r. The subarrays
A[p.. q] and A[q +1 .. r] are sorted

Postcondition: The subarray A[p..r] is sorted

Initialization: Prior to the first iteration of the loop, we have k = p; i=j=1; L[i] and R[j] contains 
smallest elements of their arrays that have not been copied back into A. The subarray A[p..k-1] is empty.

Maintenance: Suppose that L[i] ≤ R[j] and A[p...k-1] contains
k-p smallest elements. Then, L[i] is copied into A[k]. The
behaviour appropriately changes when the relationship between
L[i] and R[j] changes.

Termination: At termination, k=r+1 and A[p...r] contains the k-p
= r-p+1 smallest elements of L [n1 +1] and R [n2 +1], in sorted
order. All but the two largest have been copied back into A, and
these two largest elements are the sentinels.

// RECURSIVE(n) is
if (n=small_value)
    return ct
else
    RECURSIVE(n1)
    ...
    RECURSIVE(nr)
    some_code


n1, n2, ... nr are somevalues smaller than n but bigger than small_value


• Base Case: Prove that RECURSIVE works for n = small_value
• Inductive Hypothesis:
• Assume that RECURSIVE works correctly for n=small_value, ..., k
• Inductive Step:
• Show that RECURSIVE works correctly for n = k + 1

// Sorting 

// you can implement stable sort using standard sort function with appending idx value and custom comp

//insertion sort
stable sort 
To sort an array of size N in ascending order iterate over the array and compare the current element (key) 
to its predecessor, if the key element is smaller than its predecessor, 
compare it to the elements before. Move the greater elements one position up to make space for the swapped element.
for(int i=1;i<n;i++){
    float key=a[i];
    int j=i-1;
    while(key<a[j]&&j>=0){
        a[j+1]=a[j];
        j--;
    }
    a[j+1]=key;
}


//Bubble sort
stable sort
In this algorithm, traverse from left and compare adjacent elements and the higher one is placed at right side. 
In this way, the largest element is moved to the rightmost end at first. 
This process is then continued to find the second largest and place it and so on until the data is sorted.

Bubble sort takes minimum time (Order of n) when elements are already sorted. Hence it is best to check if the array is already sorted or not beforehand, to avoid O(N2) time complexity.

#include <bits/stdc++.h>
using namespace std;

void bubbleSort(int arr[], int n)
{
	int i, j;
	bool swapped;
	for (i = 0; i < n - 1; i++) {
		swapped = false;
		for (j = 0; j < n - i - 1; j++) {
			if (arr[j] > arr[j + 1]) {
				swap(arr[j], arr[j + 1]);
				swapped = true;
			}
		}

		// If no two elements were swapped
		// by inner loop, then break
		if (swapped == false)
			break;
	}
}

// Function to print an array
void printArray(int arr[], int size)
{
	int i;
	for (i = 0; i < size; i++)
		cout << " " << arr[i];
}

// Driver program to test above functions
int main()
{
	int arr[] = { 64, 34, 25, 12, 22, 11, 90 };
	int N = sizeof(arr) / sizeof(arr[0]);
	bubbleSort(arr, N);
	cout << "Sorted array: \n";
	printArray(arr, N);
	return 0;
}



// Selection-Sort(A[1..n])

For i = n downto 2
A: Find the largest element among A[1 ..i]
B: Exchange it with A[i]

not stable

// C++ program for implementation of
// selection sort
#include <bits/stdc++.h>
using namespace std;

// Function for Selection sort
void selectionSort(int arr[], int n)
{
	int i, j, min_idx;

	// One by one move boundary of
	// unsorted subarray
	for (i = 0; i < n - 1; i++) {

		// Find the minimum element in
		// unsorted array
		min_idx = i;
		for (j = i + 1; j < n; j++) {
			if (arr[j] < arr[min_idx])
				min_idx = j;
		}

		// Swap the found minimum element
		// with the first element
		if (min_idx != i)
			swap(arr[min_idx], arr[i]);
	}
}

// Function to print an array
void printArray(int arr[], int size)
{
	int i;
	for (i = 0; i < size; i++) {
		cout << arr[i] << " ";
		cout << endl;
	}
}

// Driver program
int main()
{
	int arr[] = { 64, 25, 12, 22, 11 };
	int n = sizeof(arr) / sizeof(arr[0]);

	// Function Call
	selectionSort(arr, n);
	cout << "Sorted array: \n";
	printArray(arr, n);
	return 0;
}




// merge sort

stable sort, more memory, not inplace, parallizable

#include<bits/stdc++.h>
using namespace std;

void merge(float a[],int l,int mid,int r){
    int left_size=mid-l+2;
    int right_size=r-mid+1;
    
    // new array for storing numbers
    float left[left_size];
    float right[right_size];
    
    
    // assigning values to new array
    for(int i=0;i<left_size-1;i++){
        left[i]=a[l+i];
    }
    for(int i=0;i<right_size-1;i++){
        right[i]=a[mid+1+i];
    }
    left[left_size-1]=2;      // very high value for last element
    right[right_size-1]=2;
    
    // pointers of left and right array
    int left_pointer=0,right_pointer=0;
    for(int i=l;i<=r;i++){
        if(left[ left_pointer ]<right[ right_pointer ]){       // whichever value is smaller, we assign 
            a[i]=left[ left_pointer ];                         // it to array
            left_pointer++;
        }
        else{
            a[i]=right[ right_pointer ];
            right_pointer++;
        }
    }
    
}

void merge_sort(float a[],int l,int r){
    if(l<r){
        // call merge sort for left half and right half of array
        int mid=(l+r)/2;
        merge_sort(a,l,mid);
        merge_sort(a,mid+1,r);
        merge(a,l,mid,r);  // merges both halfs
    }
}

Method2: Inplace Merging (TLE) [T(n) = O(n^2)]
void inPlaceMerge(vector<int> &nums, int low, int mid, int high) {
	if (low >= high) return;
	int l = low, r = mid + 1, size = high - low + 1;
	while (l <= mid and r <= high) {
		if (nums[l] <= nums[r]) l++;
		else {
			int val = nums[r];
			for (int k = r++; k > l; k--)
				nums[k] = nums[k - 1];
			nums[l++] = val;
			mid++;
		}
	}
}

void mergeSort(vector<int> &nums, int low, int high) {
	if (low >= high) return;
	int mid = (high - low) / 2 + low;
	mergeSort(nums, low, mid);
	mergeSort(nums, mid + 1, high);
	inPlaceMerge(nums, low, mid, high);
}


// Quick sort
not stable
Quick Sort, Heap Sort etc., can be made stable by also taking the position of the elements 
into consideration. This change may be done in a way that does not compromise a lot on the 
performance and takes some extra space, possibly \theta(n).


#include<bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

int partition(int *a,int l,int r){
    int key=a[r];
    int left_pointer=l;
    int right_pointer=l;
    for( ;right_pointer<r;right_pointer++){
        if(a[right_pointer]<key){
            swap(a[left_pointer],a[right_pointer]);
            left_pointer++;
        }
    }
    swap(a[left_pointer],a[right_pointer]);

    return left_pointer;

}


void quick_sort(int *a,int l,int r){
    if(l<r){
        int m=partition(a,l,r);
        quick_sort(a,l,m-1);
        quick_sort(a,m,r);
    }
}

int main(){
    
    int n=10;               
    int N=10000;
    vector<int>not_prime(N+1,-1);       // generating prime
    for(int i=2;i<=N;i++){
        if(not_prime[i]==-1&&i*i<=N){
            for(int j=i*i;j<=N;j=j+i){
                not_prime[j]=1;
            }
        }
    }
    int a[1205];
    int cnt=0;
    for(int i=2;i<N;i++){
        if(not_prime[i]==-1&&cnt<1205){
            a[cnt]=i;
            cnt++;
        }
    } 
    random_shuffle(a,a+1205);         //shuffling
    
    for(int i=0;i<n;i++){        // intial array
        cout<<a[i]<<" ";
    }cout<<endl;

    quick_sort(a,0,n-1);    // function call
 
    for(int i=0;i<n;i++){               // output array
        cout<<a[i]<<" ";
    }cout<<endl;


    
    // for(n=10;n<=1000;n=n+10){
    //     int cnt=0;
    //     auto start = chrono::high_resolution_clock::now();
    //     random_shuffle(a,a+1205);
    //     quick_sort(a,0,n-1);
        
    //     auto end = chrono::high_resolution_clock::now();
  

    //     double time_taken=chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    //     time_taken *= 1e-9;
    //     cout<<fixed<<time_taken<<setprecision(9)<<",";
    // }


    return 0;
}

#include<bits/stdc++.h>
using namespace std;

int random_partition(int *a,int l,int r){
    int total_elements=r-l+1;
    int key_index=l+(rand()%total_elements);
    swap(a[key_index],a[r]);

    int key=a[r];
    int left_pointer=l;
    int right_pointer=l;
    for( ;right_pointer<r;right_pointer++){
        if(a[right_pointer]<key){
            swap(a[left_pointer],a[right_pointer]);
            left_pointer++;
        }
    }
    swap(a[left_pointer],a[right_pointer]);

    return left_pointer;

}


void quick_sort(int *a,int l,int r){
    if(l<r){
        int m=random_partition(a,l,r);
        quick_sort(a,l,m-1);
        quick_sort(a,m,r);
    }
}

int main(){
    int n=10;
    int N=10000;
    vector<int>not_prime(N+1,-1);        //generating prime
    for(int i=2;i<=N;i++){
        if(not_prime[i]==-1&&i*i<=N){
            for(int j=i*i;j<=N;j=j+i){
                not_prime[j]=1;
            }
        }
    }
    int a[1205];
    int cnt=0;
    for(int i=2;i<N;i++){
        if(not_prime[i]==-1&&cnt<1205){
            a[cnt]=i;
            cnt++;
        }
    }
    random_shuffle(a,a+1205);          // shuffling

    for(int i=0;i<n;i++){
        cout<<a[i]<<" ";               // intial array
    }cout<<endl;
    
    quick_sort(a,0,n-1);

    for(int i=0;i<n;i++){               // output array
        cout<<a[i]<<" ";
    }cout<<endl;
    
    
    // for(n=10;n<=1000;n=n+10){
    //     int cnt=0;
    //     auto start = chrono::high_resolution_clock::now();

    //     quick_sort(a,0,n-1);
        
    //     auto end = chrono::high_resolution_clock::now();
  

    //     double time_taken=chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    //     time_taken *= 1e-9;
    //     cout<<fixed<<time_taken<<setprecision(9)<<",";
    // }


    return 0;
}


// heap sort
T(n) = T(2n/3) + O(1) in heap
differnetiate sum (1+x)^-1   to get x in front  (max heap time complexity proof)


// why comparision based sorting takes O(nlogn)
It is well known that comparison-based sorting algorithms on a single CPU require logN! time1 , w
hich is well approximated[Knuth 1981] by N logN. This limitation arises from the fact the the unsorted data 
can be in one of N! possible arrangements and that each individual comparison eliminates at most half of the 
arrangements from consideration. ~A. Tridgell “Efficient Algorithms for Sorting and Synchronization”


// bucket sort
stable sort
#include<bits/stdc++.h>
#include <chrono>
using namespace std;

void insertion_sort(vector<float>&a,int n){          // sorting buckets
    for(int i=1;i<n;i++){
        float key=a[i];
        int j=i-1;
        while(key<a[j]&&j>=0){
            a[j+1]=a[j];
            j--;
        }
        a[j+1]=key;
    }
}

int main(){
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

    int n;
    cin>>n;
    float a[n],b[n];
    for(int i=0;i<n;i++){
        a[i]=uniform_int_distribution<int>(0, 100000)(rng);
        a[i]=a[i]/100000;                                         // generating random numbers
    }

    for(int i=0;i<n;i++){
        cout<<a[i]<<" ,";                                      // printing input array
    }
    cout<<endl;
    
    int m=10;

    vector<vector<float>>v(m);

    for(int i=0;i<n;i++){
        int index=a[i]*10;
        v[index].push_back(a[i]);                    //pushing elements into buckets
    }

    for(int i=0;i<m;i++){
        insertion_sort(v[i],v[i].size());              // sorting each buckets
    }

    int j=0;
    for(int i=0;i<m;i++){                             // concatination of all buckets
        for(auto x:v[i]){
            b[j]=x;
            j++;
            cout<<x<<" ";
        }
        cout<<endl;
    }

    for(int i=0;i<n;i++){                           // printing output array
        cout<<b[i]<<" ";
    }
    cout<<endl;


    return 0;
}




//radix sort
lexo Sort first by 2nd element of tuple and then by 1st element of tuple
#include<bits/stdc++.h>
using namespace std;

int four_digit_decimal(){
    return (rand()%9+1)*1000+(rand()%10)*100+(rand()%10)*10+(rand()%10)*1;// generating 4 digit random number
}

void counting_sort(int *a,int divisor,int &n){        //counting sort implementation
    int b[n];
    for(int i=0;i<n;i++){
        b[i]=a[i];
    }

    int c[10]={0};
    for(int i=0;i<n;i++){
        int x=(a[i]/divisor)%10;
        c[x]++;
    }
    
    for(int i=1;i<10;i++){
        c[i]=c[i-1]+c[i];
    }
    
    for(int i=n-1;i>=0;i--){
        int x=(b[i]/divisor)%10;
        a[c[x]-1]=b[i];
        c[x]--;
    }
    
}

int main(){
    int n;
    cin>>n;
    int a[n];
    for(int i=0;i<n;i++){
        a[i]=four_digit_decimal();         // intializing array with random 4 digit number
    }  
    
    for(int i=0;i<n;i++){              // printing input array
        cout<<a[i]<<endl;
    }
    int p=1;
    for(int i=0;i<4;i++){
        counting_sort(a,p,n);          // applying counting sort for all digits starting from back
        p=p*10;
    }
    
    cout<<endl;

    for(int i=0;i<n;i++){            // printing output array
        cout<<a[i]<<endl;
    }

    return 0;
}

// Method2: Using Partitioning (Accepted) [T(n) = Theta(n)]
// Idea is to partition the Array with pivot as mininum Positive Element and thus, left half array is of -ve no. s 
// (if any) and right half array is of +ve no.s (if any). Finally we apply Radix Sort on both the parts.
// Note that to sort -ve no.s only with Radix Sort, we need to reverse sort the -ve no.s eg: [-5,-4,-3] and [3,4,5].


// counting sort

#include<iostream>
using namespace std;

int main(){
    int n,k;
    cin>>k;
    n=2*k;

    int a[n];
    for(int i=0;i<n;i++){               // generating random number between o to k
        a[i]=rand()%k;
    }

    int c[k]={0};
    for(int i=0;i<n;i++){               // maintaining counting array
        c[a[i]]++;
    }
    for(int i=1;i<k;i++){              // maintaing counting array as prefix sum
        c[i]=c[i-1]+c[i];
    }

    int b[n];
    for(int i=n-1;i>=0;i--){
        b[c[a[i]]-1]=a[i];               // assigning values to new array using index from count array
        c[a[i]]--;
    }

    for(int i=0;i<n;i++){              // printing input array
        cout<<a[i]<<" ";
    }
    cout<<endl;
    for(int i=0;i<n;i++){              // printing output array
        cout<<b[i]<<" ";
    }
    cout<<endl;

    return 0;
}


// huffmann coding
#include<bits/stdc++.h>
using namespace std;

struct node{                                 // data structure to store tree node
    int ascii_value;
    int frequency;
    struct node *left;
    struct node *right;

    node(int value,int freq){
        ascii_value=value;
        frequency=freq;
        left=NULL;
        right=NULL;
    }
};

struct compararator{
	bool operator()(node* left, node* right){               // cmparator to fin minimum frequency node
		return left->frequency>right->frequency;
	}
};

void huffman_coding(node *root,string huffman_str,vector<string>&ascii_huffman_coding){         // function to find huffman coding
    if(root!=NULL){ 
        if(root->ascii_value!=35){                          // if non leaf node then it our required alphabet
            ascii_huffman_coding[root->ascii_value]=huffman_str;
        }
        else{
            huffman_coding(root->left,huffman_str+"0",ascii_huffman_coding);     // otherwise we can visit their children and add 0 on left
            huffman_coding(root->right,huffman_str+"1",ascii_huffman_coding);    // and 1 on right side
        }
    }
}



int main(){

    // string s(""),line("");
    // getline(cin,line);                       // taking input
    // while(line!=""){
    //     s=s+line;
    //     getline(cin,line);
    // }
    string s="The Department of Computer Science and Engineering (CSE) was set up in July 2009. It is one of the department under the School of Engineering and offers Bachelor of Technology (BTech) and Doctor of Philosophy (PhD) programs. The department adopts a modern approach to teaching wherein students are rendered in adequate academic freedom to innovate and learn in the process. State of the art facilities including the latest software and advanced hardware are available in various laboratories for the use in both teaching and research. This facilitates adequate implementation of major B.Tech projects and for verification and validation of research results. The faculty members of the department are from diverse streams and specializations. Being a part of an emerging and relatively new institute, together with extremely competent research faculty, the Computer Science and Engineering faculty of IIT Indore offer a unique interactive platform for the students to explore the arena of fundamental and applied research. Indore is fast developing into a major IT hub in central India. Organizations like CSC and Impetus have a strong presence in the city and major players like TCS, Infosys, and Wipro are in the process of setting up their offices. For the discipline this provides an excellent opportunity to foster industry-academia partnerships.";
    
    priority_queue<node*,vector<node*>,compararator>pq;    // min heap to fin min freq alphabet
    
    int ascii_freq[256]={0};
    for(int i=0;i<s.size();i++){
        int ascii_value=(int)s[i];
        ascii_freq[ascii_value]++;                         // finding frequency of alphabets in text
    }
    for(int i=0;i<256;i++){
        if(ascii_freq[i]>0){
            pq.push(new node(i,ascii_freq[i]));            // adding alphabets to min heap
        }
    }
    
    while(pq.size()>1){
        node *first=pq.top();
        pq.pop();
        node *second=pq.top();                             // taking out first two minimum elements
        pq.pop();
        int total_freq=first->frequency+second->frequency;
        node *parent=new node(35,total_freq);              // adding their parent with alphabet '#' and freq with 
        parent->left=first;                                // sum of two min frequencies
        parent->right=second;
        pq.push(parent);
    }
    
    node *root=pq.top();                                    
    string huffman_str="";
    vector<string>ascii_huffman_coding(256,"");
    huffman_coding(root,huffman_str,ascii_huffman_coding);                       // function to print hufman coding
    
    
    vector<pair<int,int>>v;                                                     // printing huffman codes in incresing 
    for(int i=0;i<256;i++){                                                     // frequency
        if(i!=35&&ascii_freq[i]>0){
            v.push_back({ascii_freq[i],i});
        }
    }
    sort(v.begin(),v.end());
    for(int i=0;i<v.size();i++){
        char c=v[i].second;
        cout<<c<<" "<<ascii_freq[v[i].second]<<" "<<ascii_huffman_coding[v[i].second]<<endl;
    }
    
    



}
    
//https://cses.fi/problemset/task/1161
//In this problem, we're asked to find the minimum cost to divide a stick with
length x into n sticks with given lengths. It helps to work backwards; what if we start with sticks of lengths 
d1, ... ,dn and merge them into one.  HUFFMAN CODING
think if we can do using dp

// medians of medians
Choosing a Pivot
• Median-of-Medians:
• Divide the n elements into n/5 groups.
•  n/5 groups contain 5 elements each. Last group can contain n mod 5 < 5 elements.
• Determine the median of each of the groups.
• Sort each group using Insertion Sort. Pick the median from the sorted list of group elements.
• Recursively find the median x of the n/5 medians.
• Recurrence for running time (of median-of-medians):
• T(n) = O(n) + T(n/5)


linear complexity by doing things like quick sort , chosing pivot



// p-np

Np - problem with exponentialy time complexity but verifiable in poly time 
A problem x is np hard if every prob in NP is poly time reducible to X.
3sat is np hard
(x1|~x2|x3)^(x3|x5|~x6)^(x3|x6|x4)^...=1
assign values to literals 
np complete - problem both np and np hard
3 sat reducible to clique problem, so np complete