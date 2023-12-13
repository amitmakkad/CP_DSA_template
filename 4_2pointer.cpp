// Contents:-
// basic 2 pointer variations
// sliding window max 
// next permutation
// k sum
// 3 sum with 2d prefix








// count the number of triangles that you can form using these array values as edge lenght
int Solution::nTriang(vector<int> &A) {
    long long int ans=0;
    int n=A.size();
    sort(A.begin(),A.end());
    for(int k=n-1;k>=2;k--){
        int i=0;
        int j=k-1;
        while(i<j){
            long long int mini=A[i]+A[j];
            long long int maxi=A[k];
            if(mini>maxi){
                ans=(ans+j-i)%mod;
                j--;
            }
            else{
                i++;
            }
        }
    }
    return ans;
}


// Given an array A of positive integers,call a (contiguous,not necessarily distinct) 
// subarray of A good if the number of different integers in that subarray is exactly B.
// Return the number of good subarrays of A.
int ans(vector<int>&A,int k){
    int cnt=0;
    int l=0,r=0;
    map<int,int>m;
    while(r<A.size()){
        m[A[r]]++;
        while(m.size()>k){
            m[A[l]]--;
            if(m[A[l]]==0){
                m.erase(m.find(A[l]));
            }
            l++;
        }
        cnt+=r-l+1;
        r++;
    }
    return cnt;
}

int Solution::solve(vector<int> &A, int B) {
    return ans(A,B)-ans(A,B-1);
}

https://leetcode.com/problems/flip-string-to-monotone-increasing/
in one traversal 
its is actually dp, where dp[i] denotes number of flips for monotone inc upto i 
class Solution {
public:
    int minFlipsMonoIncr(string s) {
        int ans = 0, num = 0;
        for (char c : s) {
            if (c == '0') {
                ans = min(num, ans + 1);
            } else {
                ++num;
            }
        }
        return ans;
    }
};



//Given a binary array A, find the maximum sequence of continuous 1's that can be formed by replacing 
//at-most B zeroes.For this problem, return the indices of maximum continuous series of 1s in order.
vector<int> Solution::maxone(vector<int> &a, int b) {
    int i,j,k,n,m,p,t ;
    n = a.size() ;
    m = 0 ;
    int start=0 ;
    t = 0 ;
    for(i=0;i<n;i++) {
        if(a[i] == 0) m++ ;
        if(m > b) {
            while(a[start] != 0) start++ ;
            start++ ;
            m-- ;
        }
        if(t < i-start+1) {
            t = i-start+1 ;
            j = start ;
        }
    }
    vector<int> v ;
    for(i=j;i<j+t;i++) v.push_back(i) ;
    return v ;
}





// sliding window max

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k){
        deque<int> dq;
        vector<int> res;
        for (int i = 0; i < k; i++) {
            while (!dq.empty() && nums[i] >= nums[dq.back()]) {     // understand what is doing here*
                dq.pop_back();
            }
            dq.push_back(i);
        }
        res.push_back(nums[dq.front()]);

        for (int i = k; i < nums.size(); i++) {
            if(dq.front() == i - k) {
                dq.pop_front();
            }
            while (!dq.empty() && nums[i] >= nums[dq.back()]) {
                dq.pop_back();
            }

            dq.push_back(i);
            res.push_back(nums[dq.front()]);
        }
        
        return res;
    }
};

// better than method using multiset
// *We may observe that in a window, the elements that come before the largest element 
// will never be selected as the largest element of any future windows. For example, consider a window [1, 2, 3, 4, 1].
// Because the window is sliding left to right, any window with the first three elements 1, 2, and 3 would also 
// have the 4


// next permutation
public class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i + 1] <= nums[i]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}




// K sum O(n^(k-1))   we are given a set of numbers and asked if there are k of them which sum to 0
// using sorting
// using hashing 


// basic 3sum
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n=nums.size();
        vector<vector<int>>ans;
        sort(nums.begin(),nums.end());
        map<int,int>m;
        for(int i=0;i<n;i++){
            m[nums[i]]=i;
        }
        for(int i=0;i<n-2;i++){
            if(nums[i]>0) return ans;
            for(int j=i+1;j<nums.size()-1;j++){
                int req=-(nums[i]+nums[j]);
                if(m.find(req)!=m.end()&&m[req]>j){
                    ans.push_back({nums[i],nums[j],req});
                }
                j=m[nums[j]];
            }
            i=m[nums[i]];
        }
        return ans;
    }
};

// hashing
class Solution {
public:
    vector<vector<int>> kSum(vector<int>& nums, long long target, int start, int k) {
        vector<vector<int>> res;
        
        // If we have run out of numbers to add, return res.
        if (start == nums.size()) {
            return res;
        }
        
        // There are k remaining values to add to the sum. The 
        // average of these values is at least target / k.
        long long average_value = target / k;
        
        // We cannot obtain a sum of target if the smallest value
        // in nums is greater than target / k or if the largest 
        // value in nums is smaller than target / k.
        if  (nums[start] > average_value || average_value > nums.back()) {
            return res;
        }
            
        if (k == 2) {
            return twoSum(nums, target, start);
        }
    
        for (int i = start; i < nums.size(); ++i) {
            if (i == start || nums[i - 1] != nums[i]) {
                for (vector<int>& subset : kSum(nums, static_cast<long>(target) - nums[i], i + 1, k - 1)) {
                    res.push_back({nums[i]});
                    res.back().insert(end(res.back()), begin(subset), end(subset));
                }
            }
        }
                                            
        return res;
    }
	
    vector<vector<int>> twoSum(vector<int>& nums, long long target, int start) {
        vector<vector<int>> res;
        unordered_set<long long> s;
    
        for (int i = start; i < nums.size(); ++i) {
            if (res.empty() || res.back()[1] != nums[i]) {
                if (s.count(target - nums[i])) {
                    res.push_back({int(target - nums[i]), nums[i]});
                }
            }
            s.insert(nums[i]);
        }
                                             
        return res;
    }  
};

// sorting
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(begin(nums), end(nums));
        return kSum(nums, target, 0, 4);
    }
	
    vector<vector<int>> kSum(vector<int>& nums, long long target, int start, int k) {
        vector<vector<int>> res;
        
        // If we have run out of numbers to add, return res.
        if (start == nums.size()) {
            return res;
        }
        
        // There are k remaining values to add to the sum. The 
        // average of these values is at least target / k.
        long long average_value = target / k;
        
        // We cannot obtain a sum of target if the smallest value
        // in nums is greater than target / k or if the largest 
        // value in nums is smaller than target / k.
        if  (nums[start] > average_value || average_value > nums.back()) {
            return res;
        }
            
        if (k == 2) {
            return twoSum(nums, target, start);
        }
    
        for (int i = start; i < nums.size(); ++i) {
            if (i == start || nums[i - 1] != nums[i]) {
                for (vector<int>& subset : kSum(nums, static_cast<long long>(target) - nums[i], i + 1, k - 1)) {
                    res.push_back({nums[i]});
                    res.back().insert(end(res.back()), begin(subset), end(subset));
                }
            }
        }
                                            
        return res;
    }
	
    vector<vector<int>> twoSum(vector<int>& nums, long long target, int start) {
        vector<vector<int>> res;
        int lo = start, hi = int(nums.size()) - 1;
    
        while (lo < hi) {
            int curr_sum = nums[lo] + nums[hi];
            if (curr_sum < target || (lo > start && nums[lo] == nums[lo - 1])) {
                ++lo;
            } else if (curr_sum > target || (hi < nums.size() - 1 && nums[hi] == nums[hi + 1])) {
                --hi;
            } else {
                res.push_back({ nums[lo++], nums[hi--] });
            }
        }
                                                           
        return res;
    }
};





//4 sum  when we have to find 1 solution, or no repeats
map<int,pair<int,int>>m;
for(int i=n-1;i>=0;i--){
    for(int j=i-1;j>=0;j--){
        int sum=x-a[i]-a[j];
        if(m.find(sum)!=m.end()){
            cout<<i+1<<" "<<j+1<<" "<<m[sum].ff+1<<" "<<m[sum].ss+1<<endl;
            return 0;
            break;
        }
        
    }
    for(int j=i+1;j<n;j++){
        m[a[i]+a[j]]={i,j};
    }
}
cout<<"IMPOSSIBLE"<<endl;




//faster hash for 3 sum with range queries(2D prefix)
http://www.usaco.org/index.php?page=viewproblem2&cpid=994

#include <bits/stdc++.h>  
using namespace std;
using ll = long long;
using vi = vector<int>;
#define pb push_back
#define rsz resize
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
using pi = pair<int, int>;
#define f first
#define s second
#define mp make_pair

void setIO(string name = "") {  // name is nonempty for USACO file I/O
	ios_base::sync_with_stdio(0);
	cin.tie(0);  // see Fast Input & Output
	// alternatively, cin.tie(0)->sync_with_stdio(0);
	if (sz(name)) {
		freopen((name + ".in").c_str(), "r", stdin);  // see Input & Output
		freopen((name + ".out").c_str(), "w", stdout);
	}
}

#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;

int N, Q;
long long ans[5000][5000];
vector<int> A;

int main() {
	setIO("threesum");
	cin >> N >> Q;
	A.resize(N);
	for (int i = 0; i < N; ++i) cin >> A[i];

	for (int i = 0; i < N; ++i) {
		gp_hash_table<int, int> g({}, {}, {}, {}, {1 << 13});
		// initialize with certain capacity, must be power of 2
		for (int j = i + 1; j < N; ++j) {
			int res = -A[i] - A[j];
			auto it = g.find(res);
			if (it != end(g)) ans[i][j] = it->second;
			g[A[j]]++;
		}
	}
	for (int i = N - 1; i >= 0; --i)
		for (int j = i + 1; j < N; ++j)
			ans[i][j] += ans[i + 1][j] + ans[i][j - 1] - ans[i + 1][j - 1];

	for (int i = 0; i < Q; ++i) {
		int a, b;
		cin >> a >> b;
		cout << ans[a - 1][b - 1] << "\n";
	}
}

// 2pointer q qith dp (another implementation of above)
#include<bits/stdc++.h>
using namespace std;
// #define int long long


int main(){
    int n,q;
    cin>>n>>q;
    int a[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    sort(a,a+n);
    vector<vector<int>>dp(n+1,vector<int>(n+1,0));
    for(int i=0;i<n-2;i++){
        int l=i+1,r=n-1;
        while(l<r){
            int sum=a[i]+a[l]+a[r];
            if(sum==0){
                dp[i][r]++;
                if(a[l]==a[l+1]&&l+1<r){
                    dp[i][r]++;
                }
                if(a[r]==a[r-1]&&l<r-1){
                    dp[i][r-1]++;
                }
                l++;
                r--;
            }
            else if(sum>0){
                r--;
            }
            else{
                l++;
            }
        }
    }
    for (int i=n-1;i>=0;i--){
		for (int j=i+1;j<n;j++){
			dp[i][j]+=(dp[i+1][j]+dp[i][j-1]-dp[i+1][j-1]);
		}
	}
	
	while(q--){
	    int l,r;
	    cin>>l>>r;
	    l--;r--;
	    cout<<dp[l][r]<<endl;
	}
    
    return 0;
}


