// Contents:-
// Max subarray
// partitions array 
// minimum lights to activate
// interval - merge and add new interval
// virtual pointer
// spiral matrix
// permutation in array
// finding duplicates/missing in array
// some q to solve











// Max subarray sum
int ans=A[0];
int curr_sum=A[0];
for(int i=1;i<A.size();i++){
    curr_sum=max(A[i],curr_sum+A[i]);
    ans=max(ans,curr_sum);
}

// Partitions the array
// You are doing this in a complex way.
for(int i=0;i<n-1;i++)
{
    if(pre[i]==2*req) ans+=cnt;
    cnt+=(pre[i]==req);
}

You are doing this in a complex way.

// In Absolute mod q, just open mod without making any conditions

// Minimum lights to activate
https://www.interviewbit.com/problems/minimum-lights-to-activate/
int ans=0,nxt=0,n=A.size();
while(nxt<n){
    int idx=-1;
    for(int j=max(nxt-B+1,0);j<min(n,nxt+B);j++){
        if(A[j]==1){
            idx=j;
        }
    }
    if(idx==-1){
        return -1;
    }
    ans++;
    nxt=idx+B;
}
return ans;

// Merge overlap intervals
https://www.interviewbit.com/problems/merge-overlapping-intervals/
sort(A.begin(), A.end(), [](Interval l, Interval r){ return l.start<r.start; });
vector<Interval> ans;
ans.push_back(A[0]);
for(int i=1; i<A.size(); i++){
    if(A[i].start<=ans.back().end) ans.back().end = max(ans.back().end, A[i].end);
    else ans.push_back(A[i]);
}
return ans;

// adding new interval
// For (a,b) and (c,d) if max(a,c)>min(b,d) then no overlap
https://www.interviewbit.com/problems/merge-intervals/
vector<Interval> Solution::insert(vector<Interval> &intervals, Interval newInterval) {
    vector<Interval> res;
    for (Interval test : intervals) {
        if (newInterval.start > test.end) res.push_back(test);
        else if(test.start > newInterval.end) {
            res.push_back(newInterval);
            newInterval = test;
        } else if (newInterval.start <= test.end || newInterval.end >= test.start) {
            newInterval = Interval(min(test.start, newInterval.start), max(test.end, newInterval.end));
        }
    }
    res.push_back(newInterval);
    return res;
}

//Given an integer array A, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
int vir_pos=0;
for(int i=0;i<A.size();i++){
    if(A[i]!=0){
        A[vir_pos]=A[i];
        vir_pos++;
    }
}
for(int i=vir_pos;i<A.size();i++){
    A[i]=0;
}

//sort the array with one traversal
int one=0;
int zero=A.size()-1;
while(one<zero){
    while(A[one]==0&&one<zero){
        one++;
    }
    while(A[zero]==1&&one<zero){
        zero--;
    }
    if(one<zero){
        swap(A[one],A[zero]);
    }
    one++;
    zero--;
    
}
//spiral matrix 2
https://www.interviewbit.com/problems/spiral-order-matrix-ii/
vector<vector<int> > Solution::generateMatrix(int A) {
    vector< vector<int> > arr(A, vector<int> (A, 0));
    int start,end,i;
    long long int x;
    x=1;
    for(start=0,end=A-1;start<=end;start++,end--)
    {
        for(i=start;i<=end;i++) { arr[start][i]=x; x++;}
        for(i=start+1;i<=end;i++) { arr[i][end]=x; x++;}
        for(i=end-1;i>=start;i--) { arr[end][i]=x; x++;}
        for(i=end-1;i>=start+1;i--) { arr[i][start]=x; x++;}
    }
    return arr;
}



// you have to find any permutation of first n positive integer that satisfy the given input string. D means the next number is smaller, while I means the next number is greater.
https://www.interviewbit.com/problems/find-permutation/
vector<int> Solution::findPerm(const string A, int B) {
    
    vector<int> ans;
    int l=1,r=B;
    for(int i=0;i<A.length();i++)
    {
       if(A[i]=='I') {ans.push_back(l);l++;}
       else {ans.push_back(r);r--;} 
    }
    ans.push_back(r);
    
    return ans;
    
}

// Single duplicate in array:-
https://www.interviewbit.com/problems/find-duplicate-in-array/
// xor
// slow-fast
// freq array
// bucket od 1 to sqrt(n)
// negative marking https://leetcode.com/problems/find-the-duplicate-number/

// Given an unsorted array Arr of size N of positive integers. One number 'A' from set {1, 2,....,N} is missing
// and one number 'B' occurs twice in array. Find these two numbers
https://practice.geeksforgeeks.org/problems/find-missing-and-repeating2512/1
// 1 sorting
// 2 freq array
// 3 use map
// 4 sum of 1..n and sum of squares
Method 5 (Use elements as Index and mark the visited places)
Approach: 
Traverse the array. While traversing, use the absolute value of every element as an index and make the value 
at this index as negative to mark it visited. If something is already marked negative then this is the
 repeating element. To find missing, traverse the array again and look for a positive value.

Method 6 (Make two equations)
Approach:

Let x be the missing and y be the repeating element.
Get the sum of all numbers using formula S = n(n+1)/2 - x + y
Get product of all numbers using formula P = 1*2*3*...*n * y / x
The above two steps give us two equations, we can solve the equations and get the values of x and y.

Method 7: (Placing every element in its correct position)
Approach: It is clear from the observation that if we sort an array, then arr[i] == i+1.
If all elements in an array satisfy this condition, means this is an ideal case. 
So the idea is to sort the given array and traverse on it and check 
if arr[i] == i + 1 if so then increment i (because this element is at its correct position), 
otherwise place this element (arr[i]) at its correct position (arr[arr[i] - 1]) 
by swapping the arr[i] and arr[arr[i] -1]. This swapping will put the element arr[i] at its correct position
(i.e arr[arr[i]-1]). After doing this operation gets over for all elements of given array then again 
traverse over the array and check if arr[i] != i + 1 if so, then this is the duplicate element and i + 1 
is the missing element.

#include <bits/stdc++.h>
using namespace std;

void getTwoElements(int arr[], int n)
{
    int repeating(0), missing(0);

    int i = 0;

    // Traverse on the array
    while (i < n) 
    {
      
        // If the element is on its correct position
        if (arr[i] == arr[arr[i] - 1])
            i++;
          // If it is not at its correct position  
        // then palce it to its correct position
          else
            swap(arr[i], arr[arr[i] - 1]);
    }

    // Find repeating and missing
    for (int i = 0; i < n; i++) {
      
        // If any element is not in its correct position
        if (arr[i] != i + 1) {
            repeating = arr[i];
            missing = i + 1;
            break;
        }
    }

    // Print answer
    cout << "Repeating: " << repeating << endl
         << "Missing: " << missing << endl;
}

// Driver code
int main()
{
    int arr[] = { 2, 3, 1, 5, 1 };
    int n = sizeof(arr) / sizeof(int);

    getTwoElements(arr, n);

    return 0;
}


Method 8 (Use XOR)

Approach:

Let x and y be the desired output elements.
Calculate XOR of all the array elements.
xor1 = arr[0]^arr[1]^arr[2].....arr[n-1]

XOR the result with all numbers from 1 to n
xor1 = xor1^1^2^.....^n

In the result xor1, all elements would nullify each other except x and y. All the bits that are set in
xor1 will be set in either x or y. So if we take any set bit (We have chosen the rightmost set bit in code) 
of xor1 and divide the elements of the array in two sets – one set of elements with the same bit set 
and other set with same bit not set. By doing so, we will get x in one set and y in another set. 
Now if we do XOR of all the elements in first set, we will get x, and by doing same in other set we will get y. 




class Solution {
public:
    int store(vector<int>&nums,int curr){
        if(curr==nums[curr]){
            return curr;
        }
        int nxt=nums[curr];
        nums[curr]=curr;
        return store(nums,nxt);
    }
    int small_equal(vector<int>&nums,int curr){
        int cnt=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]<=curr){
                cnt++;
            }
        }
        return cnt;
    }
    int findDuplicate(vector<int>& nums) {
        int n=nums.size();

        // 1st sort array

        // 2nd set

        // 3rd negative marking
        // for(int i=0;i<n;i++){
        //     int curr=abs(nums[i]);
        //     if(nums[curr]<0){
        //         return curr;
        //     }
        //     nums[curr]=-nums[curr];
        // }
        // for(int i=0;i<n;i++){
        //     nums[i]=abs(nums[i]);
        // }

        // 4th hashmap recursive
        // return store(nums,0);

        // 5th hashmap iterative
        // while(nums[0]!=nums[nums[0]]){
        //     swap(nums[0],nums[nums[0]]);
        // }
        // return nums[0];

        // 6th b search 
        // int l=1,r=n;
        // int ans=0;
        // while(l<=r){
        //     int mid=(l+r)/2;
        //     if(small_equal(nums,mid)>mid){
        //         ans=mid;
        //         r=mid-1;
        //     }
        //     else{
        //         l=mid+1;
        //     }
        // }
        // return ans;

        // 7th Bitwise
        // int ans=0;
        // for(int j=0;j<32;j++){
        //     int mask=(1<<j);
        //     int exp=0,curr=0;
        //     for(int i=0;i<n;i++){
        //         if(i&mask) exp++;
        //         if(nums[i]&mask) curr++;
        //     }
        //     if(curr>exp){
        //         ans=(ans|mask);
        //     }
        // }
        // return ans;

        // 8th floyd 
        int slow=nums[0];
        int fast=nums[0];
        do{
            slow=nums[slow];
            fast=nums[nums[fast]];
        }while(slow!=fast);

        slow=nums[0];
        while(slow!=fast){
            slow=nums[slow];
            fast=nums[fast];
        }
        return slow;

        return 0;
    }
};








// C++ program to Find the repeating
// and missing elements

#include <bits/stdc++.h>
using namespace std;

/* The output of this function is stored at
*x and *y */
void getTwoElements(int arr[], int n,
                    int* x, int* y)
{
    /* Will hold xor of all elements 
    and numbers from 1 to n */
    int xor1;

    /* Will have only single set bit of xor1 */
    int set_bit_no;

    int i;
    *x = 0;
    *y = 0;

    xor1 = arr[0];

    /* Get the xor of all array elements */
    for (i = 1; i < n; i++)
        xor1 = xor1 ^ arr[i];

    /* XOR the previous result with numbers 
    from 1 to n*/
    for (i = 1; i <= n; i++)
        xor1 = xor1 ^ i;

    /* Get the rightmost set bit in set_bit_no */
    set_bit_no = xor1 & ~(xor1 - 1);

    /* Now divide elements into two 
    sets by comparing a rightmost set
    bit of xor1 with the bit at the same 
    position in each element. Also, 
    get XORs of two sets. The two
    XORs are the output elements. 
    The following two for loops 
    serve the purpose */
    for (i = 0; i < n; i++) {
        if (arr[i] & set_bit_no)
            /* arr[i] belongs to first set */
            *x = *x ^ arr[i];

        else
            /* arr[i] belongs to second set*/
            *y = *y ^ arr[i];
    }
    for (i = 1; i <= n; i++) {
        if (i & set_bit_no)
            /* i belongs to first set */
            *x = *x ^ i;

        else
            /* i belongs to second set*/
            *y = *y ^ i;
    }

    /* *x and *y hold the desired
        output elements */
}

/* Driver code */
int main()
{
    int arr[] = { 1, 3, 4, 5, 5, 6, 2 };
    int* x = (int*)malloc(sizeof(int));
    int* y = (int*)malloc(sizeof(int));
    int n = sizeof(arr) / sizeof(arr[0]);

    getTwoElements(arr, n, x, y);
    cout << " The missing element is " << *x << " and the repeating"
         << " number is " << *y;
    getchar();
}


https://www.interviewbit.com/problems/first-missing-integer/
https://www.interviewbit.com/problems/set-intersection/
https://www.interviewbit.com/problems/maximum-consecutive-gap/

int Solution::maximumGap(const vector<int> &num) {
    if(num.empty() || num.size() < 2)
        return 0;
    int maxNum = *max_element(num.begin(), num.end());
    int minNum = *min_element(num.begin(), num.end());
    //average gap from minNum to maxNum.
    int gap = (maxNum - minNum - 1) / (num.size() - 1) + 1;

    //number of buckets = num.size() - 1
    vector<int> bucketsMin(num.size() - 1, INT_MAX);
    vector<int> bucketsMax(num.size() - 1, INT_MIN);
    //put into buckets
    for (int i = 0; i < num.size(); i++) {
        if (num[i] != maxNum && num[i] != minNum)
        {
            int buckInd = (num[i] - minNum) / gap;
            bucketsMin[buckInd] = min(bucketsMin[buckInd], num[i]);
            bucketsMax[buckInd] = max(bucketsMax[buckInd], num[i]);
        }
    }
    int maxGap = INT_MIN;
    int previous = minNum;
    for (int i = 0; i < num.size() - 1; i++) {
        if (bucketsMin[i] == INT_MAX && bucketsMax[i] == INT_MIN)
            continue;   //empty
        //i_th gap is minvalue in i+1_th bucket minus maxvalue in i_th bucket 
        maxGap = max(maxGap, bucketsMin[i] - previous);
        previous = bucketsMax[i];
    }
    maxGap = max(maxGap, maxNum - previous);
    return maxGap;
}



https://www.interviewbit.com/problems/reorder-data-in-log-files/
https://www.interviewbit.com/problems/n3-repeat-number/


sort 
map/set 
visited/count array 
slow fast 
xor 
index sort recursive/iterative
b search
visited with negative num
bitwise
equations
buckets