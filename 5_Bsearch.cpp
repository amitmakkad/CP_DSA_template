// when l,r becomes negative than that formula may not work
https://codeforces.com/contest/1223/problem/C

// briliant way of applying b search which you never thought   (a[i]>mid then b[i]=1 else b[i]=-1)
https://codeforces.com/problemset/problem/1486/D 
You are a given an array a of length n. Find a subarray a[l..r] with length at least k with the largest median.
apply b search to check whether mid is median of subarray , to check subarray of lenght atleast k
convert array into 1 if a[i]>=mid else -1 
now if you find a subarray whose sum>0 with lenght >k exists then possible
this is possible with 2nd way of kadane algorithm


// cost function analysis - bsearch on mid-1, mid, mid+1 types 

// b search not always work
https://leetcode.com/problems/fair-distribution-of-cookies/ 


// applications of b search
https://www.geeksforgeeks.org/check-if-value-exists-in-level-order-sorted-complete-binary-tree/

// To do b search on roation of array 


https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/ same as above candies distribution
b search in different way

class Solution {
public:
    
    bool is_possible(vector<int>& arr, int i, int n, int mid, vector<int>& worker, int k)
    {
        // base case
        
        if(i == n)
            return true;
        
        // assign the curr_work to the workers which can do the curr_work
        
        for(int j = 0; j < k; j++)
        {
            if(worker[j] + arr[i] <= mid)
            {
                worker[j] += arr[i];
                
                if(is_possible(arr, i + 1, n, mid, worker, k))
                    return true;
                
                worker[j] -= arr[i];
            }
            
            // if any of the worker doesn't get any job then this combination will not produce optimal result
            
            if(worker[j] == 0) // tle without this
                break;
        }
        
        return false;
    }
    
    int minimumTimeRequired(vector<int>& arr, int k) {
        
        int n = arr.size();
        
        // sort the array in descending order
        
        sort(arr.begin(), arr.end(), greater<int> ());
        
        int mini = INT_MAX;
        
        // apply binary search on search space {low, high}
        
        int low = *max_element(arr.begin(), arr.end());
            
        int high = 0;
            
        for(int i = 0; i < n; i++)
        {
            high += arr[i];
        }
            
        int ans = -1;
        
        // for every mid, distribute the works into k worker then take min. of ans
            
        while(low <= high)
        {
            int mid = low + (high - low) / 2;
            
            vector<int> worker(k, 0);
                
            if(is_possible(arr, 0, n, mid, worker, k))
            {
                ans = mid;
                    
                high = mid - 1;
            }
            else
            {
                low = mid + 1;
            }
        }
            
        mini = min(mini, ans); 
        
        return mini;
    }  
};


https://leetcode.com/problems/find-in-mountain-array/ why you confused
first find peak, then do b search and reverse b search


Given an array A[0 … n-1] containing n positive integers, a subarray A[i … j] is bitonic 
if there is a k with i <= k <= j such that A[i] <= A[i + 1] … = A[k + 1] >= .. A[j - 1] > = A[j]

// below 3 are np compete, dfs followed not binary search
https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/
https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
https://leetcode.com/problems/fair-distribution-of-cookies/


https://www.interviewbit.com/problems/red-zone/
https://www.interviewbit.com/problems/median-of-array/
https://www.interviewbit.com/problems/simple-queries/
