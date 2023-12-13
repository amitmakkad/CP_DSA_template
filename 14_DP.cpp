// Contents:-
// dp basics
// Longest repeating subsequence
// basic qs
// longest valid parenthesis many approaches
// bsearch in egg drop
// knapsack
// a diff dp using helper func 
// path on grid diff view
// coin problem
// lis 
// bitmask dp and n!
// digit dp
// range dp
// fill triplets 
// some q

// what is dp?
// Optimal Substructure Property if the optimal solution of the given problem can be obtained by using 
// the optimal solution to its subproblems instead of trying every possible way to solve the subproblems. 

// Overlapping subproblem
// Like Divide and Conquer, Dynamic Programming combines solutions to sub-problems.
// Dynamic Programming is mainly used when solutions to the same subproblems are needed again and again.


//is subsequence
dont always think dp of standard q 
bool isSubsequence(string s, string t) {
    int j=0;
    for(int i=0;i<t.size()&&j<s.size();i++){
        if(t[i]==s[j]){
            j++;
        }
    }
    return j==s.size();
}









// to find longest repeating subsequence, 
lets try finding the longest common subsequence between the string A
and itself ( LCS(A, A) ). The only restriction we want to impose is that you cannot match a character 
with its replica in the other string.


//Given two strings A and B of the same length, determine if B is a scrambled string of S.
//Given a string A, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.
bool isScramble(int a,int b,int c,int d,string s1,string s2)
{
    if(b-a != d-c)
        return false;
    if(a==b && c==d)
    {
        if(s1[a]==s2[c])
            return true;
        return false;
    }
    string temp1=s1.substr(a,b-a+1);
    string temp2=s2.substr(c,d-c+1);
    sort(temp1.begin(),temp1.end());
    sort(temp2.begin(),temp2.end());
    if(temp1!=temp2)
        return false;
    
    for(int i = 1; i < s1.length(); i++) { // i being the root position
        if(isScramble(s1.substring(0,i), s2.substring(0,i)) && isScramble(s1.substring(i), s2.substring(i))) return true;
        if(isScramble(s1.substring(0,i), s2.substring(s2.length()-i)) && isScramble(s1.substring(i), s2.substring(0, s2.length()-i))) return true;
    }
    
}
int Solution::isScramble(const string A, const string B) {
}

https://leetcode.com/problems/generate-parentheses/
unordered_map<int, unordered_set<string>> mp;
    
    vector<string> generateParenthesis(int n) {
        mp[1].insert("()");
        for (int i = 2; i <= n; ++i) {
            for (auto &prev : mp[i - 1]) {
                for (int j = 0; j < prev.size(); ++j) {
                    mp[i].insert(prev.substr(0, j) + "()" + prev.substr(j));
                }
            }
        }
        
        return {mp[n].begin(), mp[n].end()};
    }


//Given three prime numbers A, B and C and an integer D. You need to 
//find the first(smallest) D integers which only have A, B, C or a combination of them as their prime factors.
vector<int> Solution::solve(int A, int B, int C, int D) {
    vector<int>ans;
    ans.push_back(1);
    int i=0,x=0,y=0,z=0;
    while(i < D){
        int tmp = min({A*ans[x],B*ans[y], C*ans[z]});
        ans.push_back(tmp);
        if(tmp == A*ans[x]) ++x;
        if(tmp == B*ans[y]) ++y;
        if(tmp == C*ans[z]) ++z;
        i++;
    }
    ans.erase(ans.begin());
    return ans;
}

// Given an integer A you have to find the number of ways to fill a 3 x A board with 2 x 1 dominoes.
int Solution::solve(int A) {
    int n=A+1;
    vector<int>a(100005,0);
    vector<int>b(100005,0);
    b[1]=1;
    a[0]=1;
    int mod=1000000007;
    for(int i=2;i<n;i++){
        a[i]=(a[i-2]+(2*b[i-1])%mod)%mod;
        b[i]=(a[i-1]+b[i-2])%mod;
    }
    return (a[n-1])%mod;
}


//Find the longest Arithmetic Progression in an integer array A of size N, and return its length.
//Let dp[i][j] be the length of Longest Arithmetic progression that ends in positions i and j, 
//i.e. last element is A[j] and element before last is A[i]. 
int Solution::solve(const vector<int> &A) {
    int n = A.size();
    if(n <= 2) {
        return n;
    }
    vector<vector<int>> dp(n, vector<int>(n, 2));
    unordered_map<int, int> pos;
    int max_len = 2;
    for(int i = 0; i < n - 1; i++) {
        for(int j = i + 1; j < n; j++) {
            int need = 2 * A[i] - A[j];
            if(pos.count(need)) {
                dp[i][j] = max(dp[i][j], dp[pos[need]][i] + 1);
            }
            max_len = max(dp[i][j], max_len);
        }
        pos[A[i]] = i;
    }
    return max_len;
}

// Given a matrix M of size nxm and an integer K, find the maximum element in the K manhattan distance 
// neighbourhood for all elements in nxm matrix. In other words, for every element M[i][j] 
// find the maximum element M[p][q] such that abs(i-p)+abs(j-q) <= K. If you have answer 
// for (K-1)th manhtaan distance of all the elements, can you use those values to find the answer 
// for Kth manhattan distance.

dp[k][i][j] = ans. for kth manhattan distance for element (i,j)
dp[k+1][i][j] = max(dp[k][i-1][j], dp[k][i+1][j], dp[k][i][j-1], dp[k][i][j+1], dp[k][i][j] )




https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
int maxProfit(vector<int>& prices) {
    int mini=prices[0];
    int ans=INT_MIN;
    for(int i=1;i<prices.size();i++){
        ans=max(ans,prices[i]-mini);
        mini=min(prices[i],mini);
    }
    return max(ans,0);
    
}
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
The profit is the sum of sub-profits. Each sub-profit is the difference between selling at day j, 
and buying at day i (with j > i). The range [i, j] should be chosen so that the sub-profit is maximum:
Actually, the range [3, 2, 5, 8, 1, 9] will be splitted into 2 sub-ranges [3, 2, 5, 8] and [1, 9].

int maxProfit(vector<int>& prices) {
    int n = prices.size();
    int m = prices[0];
    int ans = 0;
    for(int i=1; i<n; i++){
        if(prices[i]<=m){
            m = prices[i];
        }
        else{
            ans += prices[i] - m;
            m = prices[i];
        }
    }
    return ans;
}
or 
int maxProfit(vector<int>& prices) {
    int maxProfit=0;
    for(int i=1; i<prices.size(); ++i) 
        maxProfit += max(prices[i] - prices[i-1], 0);

    return maxProfit;
}
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
public int maxProfit(int[] prices) {
    int hold1 = Integer.MIN_VALUE, hold2 = Integer.MIN_VALUE;
    int release1 = 0, release2 = 0;
    for(int i:prices){                              // Assume we only have 0 money at first
        release2 = Math.max(release2, hold2+i);     // The maximum if we've just sold 2nd stock so far.
        hold2    = Math.max(hold2,    release1-i);  // The maximum if we've just buy  2nd stock so far.
        release1 = Math.max(release1, hold1+i);     // The maximum if we've just sold 1nd stock so far.
        hold1    = Math.max(hold1,    -i);          // The maximum if we've just buy  1st stock so far. 
    }
    return release2; ///Since release1 is initiated as 0, so release2 will always higher than release1.
}

approach 1 
We can find all adjacent valley/peak pairs and calculate the profits easily. 
Instead of accumulating all these profits like Buy&Sell Stock II, we need the highest k ones.

The key point is when there are two v/p pairs (v1, p1) and (v2, p2),
satisfying v1 <= v2 and p1 <= p2, we can either make one transaction at [v1, p2], 
or make two at both [v1, p1] and [v2, p2]. The trick is to treat [v1, p2] as the first transaction, 
and [v2, p1] as the second. Then we can guarantee the right max profits in both situations, 
p2 - v1 for one transaction and p1 - v1 + p2 - v2 for two.
Finding all v/p pairs and calculating the profits takes O(n) since there are up to n/2 such pairs. 
And extracting k maximums from the heap consumes another O(klgn).

approach 2 (reuse last idea)
public int maxProfit(int k, int[] prices) {        
    int[] buy = new int[k + 1], sell = new int[k + 1];
    Arrays.fill(buy, Integer.MIN_VALUE);
    for (int price : prices) {
        for (int i = 1; i <= k; i++) {
            buy[i] = Math.max(buy[i], sell[i - 1] - price);
            sell[i] = Math.max(sell[i], buy[i] + price);
        }
    }
    return sell[k];
}
appraoch 3
 * dp[i, j] represents the max profit up until prices[j] using at most i transactions. 
 * dp[i, j] = max(dp[i, j-1], prices[j] - prices[jj] + dp[i-1, jj]) { jj in range of [0, j-1] }
 *          = max(dp[i, j-1], prices[j] + max(dp[i-1, jj] - prices[jj]))
 * dp[0, j] = 0; 0 transactions makes 0 profit
 * dp[i, 0] = 0; if there is only one price data point you cant make any transaction.

int[][] dp = new int[k+1][n];
for (int i = 1; i <= k; i++) {
    int localMax = dp[i-1][0] - prices[0];
    for (int j = 1; j < n; j++) {
        dp[i][j] = Math.max(dp[i][j-1],  prices[j] + localMax);
        localMax = Math.max(localMax, dp[i-1][j] - prices[j]);
    }
}
return dp[k][n-1];

my code:-
//dp[i][j] = maximum profit from most i transactions using prices[0..j]
int Solution::solve(vector<int> &A, int B) {
    int n=A.size();
    B=min(B,n);
    vector<vector<int>>dp(B+1,vector<int>(n,0));
    for(int i=1;i<=B;i++){
        int diff=dp[i-1][0]-A[0];
        for(int j=1;j<n;j++){
            diff=max(diff,dp[i-1][j]-A[j]);
            dp[i][j]=max(dp[i][j-1],diff+A[j]);
        }
    }
    return dp[B][n-1];
}

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/solutions/900051/fully-explained-all-buy-and-sell-problems-c-recursive-memoization-minor-difference/
dp approach to all 4 types


//Longest valid Parentheses
see how beautifully it uses previous state

dp[i]=dp[i-2]+2                                    // you forgot to consider dp[i-2]
dp[i]=dp[i-1]+dp[i-dp[i-1]-2]+2                    // .......                dp[i-dp[i-1]-2]

public class Solution {
    public int longestValidParentheses(String s) {
        int maxans = 0;
        int dp[] = new int[s.length()];
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = Math.max(maxans, dp[i]);
            }
        }
        return maxans;
    }
}
// to do here other approaches



//egg dropping problem
Binary search


//Given an integer array A of size N. You have to merge all the elements of the array into one with the 
//minimum possible cost.
//The rule for merging is as follows:
//Choose any two adjacent elements of the array with values say X and Y and merge them into a single element
//with value (X + Y) paying a total cost of (X + Y).
//Return the minimum possible cost of merging all elements.

This problem can be solved using dynamic programming.
Let s define the states of the dp first. 
DP[l][r] will be the minimum cost of merging the subarray A[l…r] into one.
Now, lets look at the recurrence relation:

DP[l][r] = min(S(l,l)+S(l+1,r)+DP[l][l]+DP[l+1][r],
                S(l,l+1)+S(l+2,r)+DP[l][l+1]+DP[l+2][r], …,
                
                S(l,r-1)+S(r,r)+DP[l][r-1]+DP[r][r])

=S(l,r)+min(DP[l][l]+DP[l+1][r],DP[l][l+1]+DP[l+2][r], …,DP[l][r-1]+DP[r][r])

//think hofmann encoding

//knacpsack
int Solution::solve(vector<int> &values, vector<int> &weights, int C) {
    int n=values.size();
    int dp[1005][1005]={0};

    for(int i=1;i<=n;i++){
        for(int j=1;j<=C;j++){
            dp[i][j]=dp[i-1][j];
            if(j-weights[i-1]>=0)
                dp[i][j]=max(dp[i][j],dp[i-1][j-weights[i-1]]+values[i-1]);
        }
    }
    return dp[n][C];
}
ex-Given an array A of positive elements, you have to flip the sign of some of its elements 
such that the resultant sum of the elements of array should be minimum non-negative(as close to zero as possible).

remember space optimized kanpsack O(n)


//Given an integer A, how many structurally unique BST’s (binary search trees) exist that can store values 1…A?
2nCn/(n+1)
or can dp logic -
If we choose the number j as the root, j - 1 numbers fall in the left subtree, n - j numbers fall in the 
right subtree. We already know how many ways there are to forming j - 1 trees using j - 1 numbers and n -j numbers.
So we add number(j - 1) * number(n - j) to our solution.

// You are given N potions arranged in a row in the form of an array A.
// Each potion has one of 100 different colors (colors have numbers from 0 to 99). 
// You need to mix all these potions together.At each step, you are going to take two potions that
// stand next to each other and mix them together,and put the resulting mixture in their place.When mixing 
// two mixtures of colors X and Y, the resulting mixture will have the color (X + Y) mod 100.
// Also, there will be some smoke in the process. The amount of smoke generated when mixing two mixtures of 
// colors X and Y is X * Y.
// Find out what is the minimum amount of smoke that you can get when mixing all the potions together.

vector<int> helper(int i, int j, vector<int> &A, vector<vector<vector<int>>> &dp){
    vector<int> res;
    if(dp[i][j][0] != -1)
        return dp[i][j];

    if(i == j){
        res.push_back(0);
        res.push_back(A[i]);
        dp[i][j] = res;
        return res;
    }
    else if(j == i+1){
        res.push_back(A[i]*A[j]);
        res.push_back((A[i]+A[j])%100);
        dp[i][j] = res;
        return res;
    }
    int min_cost = INT_MAX;
    for(int k = i+1; k <= j; ++k){
        vector<int> v1 = helper(i, k-1, A, dp);
        vector<int> v2 = helper(k, j, A, dp);
        int c1 = v1[0];
        int c2 = v2[0];
        int num1 = v1[1];
        int num2 = v2[1];
        int cost = c1 + c2 + num1*num2;
        int num = (num1 + num2)%100;
        vector<int> v;
        v.push_back(cost);
        v.push_back(num);
        if(cost < min_cost){
            min_cost = cost;
            res = v;
        }
    }
    dp[i][j] = res;
    return res;
}
int Solution::minSmoke(vector<int> &A) {
    
    int n = A.size();
    if(n == 1)
        return 0;
    else if(n == 2)
        return A[0]*A[1];
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(2, -1)));
    vector<int> ans;
    ans = helper(0, n-1, A, dp);
    return ans[0];
}

// You know that an array has n integers between 1 and m and the absolute difference between 
// two adjacent values is at most 1
// Given a description of the array where some values may be unknown, 
// your task is to count the number of arrays that match the description.
// it is path on grids*********
    int n,m;
    cin>>n>>m;
    int a[n];
    vector<vector<int>>dp(n+1,vector<int>(m+1,0));
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    if(a[0]==0){
        for(int i=1;i<=m;i++){
            dp[0][i]=1;
        }
    }
    else{
        dp[0][a[0]]=1;
    }
    for(int i=1;i<n;i++){
        if(a[i]==0){
            for(int j=1;j<=m;j++){
                dp[i][j]+=dp[i-1][j];
                if(j-1>0) dp[i][j]+=dp[i-1][j-1];
                if(j+1<=m) dp[i][j]+=dp[i-1][j+1];
                dp[i][j]=dp[i][j]%mod;
            }
            
        }
        else{
            int j=a[i];
            dp[i][j]+=dp[i-1][j];
            if(j-1>0) dp[i][j]+=dp[i-1][j-1];
            if(j+1<=m) dp[i][j]+=dp[i-1][j+1];
            dp[i][j]=dp[i][j]%mod;
        }
    }
    int ans=0;
    for(int i=1;i<=m;i++){
        ans=(ans+dp[n-1][i])%mod;
    }
    cout<<ans<<endl;

//in coin problems or number of ways
for(int i=1;i<n;i++){
    for(int j=300000-1;j>=i;j--){   // counting should be reversed, now from 0 to n
        dp[j]=(dp[j]+dp[j-i])%mod;
    }
    
}

// lis[i] has 0..i ending at i, not till i becarefull of that


//lis in nlog n
//ans is stored in size of vector or set

//using vector
    int n;
    cin>>n;
    int a[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    vector<int> v;
    for (int i = 0; i < n; i++) {
        auto it = lower_bound(v.begin(), v.end(), a[i]);
        if (it != v.end()) *it = a[i];
        else v.push_back(a[i]);
    }
    cout<< v.size()<<endl;
    

//using set
    set<int> S;
    set<int>::iterator it;
    for (int x : a)
    {
        S.insert(x);
        it = S.upper_bound(x);
        if (it != S.end()) S.erase(it);
    }
    
    cout << S.size();

//using segment tree
    int n;
    cin>>n;
    int a[n];
    map<int,int>m;
    for(int i=0;i<n;i++){
        cin>>a[i];
        m[a[i]]++;
    }
    int cnt=1;
    for(auto &x:m){
        x.second=cnt;
        cnt++;
    }
    for(int i=0;i<n;i++){
        a[i]=m[a[i]];
    }
    
    int b[n+1]={0};
    int tree[4*(n+1)]={0};
    
    build_max(b,tree,1,0,n);
    
    int ans=0;
    for(int i=0;i<n;i++){
        int x=query_max(b,tree,1,0,n,1,a[i]-1);
        update(b,tree,1,0,n,a[i],x+1);
        ans=max(x+1,ans);
    }
    cout<<ans<<endl;
    


// sometimes questions are much related to lis just make pair in sorted order and apply lis in other dim
First, what can we say about two segments (l1, r1) and (l2, r2) if they intersect (assuming l1 < l2)
Since these segments are straight, notice how l1 < l2 implies r1 > r2

This means that a set of non-intersecting segments satisfies li < lj implies ri < rj for all pairs (i, j)

Let A be an array where A[i] = x means that the segment with its right
endpoint at position i has its left endpoint at position x.

If we were asked to find the maximum size of a set of non-intersecting segments,
the answer would be the LIS of A!

The minimum number of increasing subsequences required to
cover A is at least the size of longest non-increasing subsequence of A.


/**
 * Notice if we have ranges [l1, r1] and [l2, r2] with l1 < l2 and r1 < r2,
 * that is, neither range is contained in the other, then
 * these cows will never collide.
 * 
 * We need to find the maximum number of ranges continuously contained
 * l1 <= l2 <= ... <= lk  <=  rk <= ... <= r1
 * 
 * Sort based on end-time DESC
 * 
 * Finding l1 <= l2 implies r1 >= r2 due to sorting
 * l1 <= l2 <= l3 <= ... <= lk implies r1 >= r2 >= r3 >= ... >= rk
 * 
 * To ensure l1 <= l2 if r2 = r1, secondary sort based on start-time ASC
 * Now, we need to find a Largest Non-Decreasing Subsequence of start-times
 */


// typical bitmask dp
elevator problem, dp stores no of rides till now and max unfinished weight
dont think much in bitmask about anything which you can do greedily

    int n,x;
    cin>>n>>x;
    int w[n];
    for(int i=0;i<n;i++){
        cin>>w[i];
    }
    vector<pair<int,int>>dp((1<<n),{n+1,x+1});
    dp[0]={1,0};
    for(int mask=1;mask<(1<<n);mask++){
        for(int i=0;i<n;i++){
            if(mask&(1<<i)){
                int prev=mask^(1<<i);
                int prev_w=dp[prev].second;
                int prev_r=dp[prev].first;
                if(w[i]+prev_w<=x){
                    prev_w+=w[i];
                }
                else{
                    prev_r++;
                    prev_w=w[i];
                }
                dp[mask]=min(dp[mask],{prev_r,prev_w});
            }
        }
    }
    cout<<dp[(1<<n)-1].first<<endl;

// n! type with dp with bitmask
https://www.interviewbit.com/problems/shortest-common-superstring/

dp[i][mask] = Optimal solution for set of strings corresponding to 1s in the mask 
                where the last added string to our solution is i-th string.

Recurrence:
dp[i][mask] = min(dp[x][mask ^ (1«i)] where {mask | (1«x) = 1} )


//digit dp
https://leetcode.com/problems/count-of-integers/    
class Solution {
public:
    int mod=1e9+7;
    int dp[23][2][2][401];

    int solve(int i,bool t1,bool t2,int sum,string &num1,string &num2){
        if(sum<0) return 0;
        if(i==num2.size()) return 1;
        if(dp[i][t1][t2][sum]!=-1) return dp[i][t1][t2][sum];
        int l=0,r=9;
        if(t1==true) l=num1[i]-'0';
        if(t2==true) r=num2[i]-'0';

        int cnt=0;
        for(int idx=l;idx<=r;idx++){
            cnt=(cnt+solve(i+1,t1&(idx==l),t2&(idx==r),sum-idx,num1,num2))%mod;
        }
        return dp[i][t1][t2][sum]=cnt%mod;
    }

    int count(string num1, string num2, int min_sum, int max_sum) {
        memset(dp,-1,sizeof(dp));
        int leadingzeros=num2.size()-num1.size();
        string num1_ext=string(leadingzeros,'0')+num1;

        int total=solve(0,1,1,max_sum,num1_ext,num2);
        int rem=solve(0,1,1,min_sum-1,num1_ext,num2);
        int ans=(total-rem+mod)%mod;
        return ans;
        
    }
};

//Range DP  nice transition tables are made
https://cses.fi/problemset/task/1080     
https://codeforces.com/problemset/problem/607/B  

https://www.interviewbit.com/problems/ways-to-color-a-3xn-board/   
// nice way of taking permutation from dp
#define mod 1000000007
#define ll long long

struct triplet{
	int x, y, z;
};

vector<triplet> trip;  //vector of 36 triplets

void fillTriplets(){
	//trip vector contain the triplets of color that can be used to paint a certain column
       trip.clear();
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			for(int k=0; k<4; k++){
				if(i!=j && j!=k) trip.push_back({i,j,k});
			}
		}
	}
}

int dp[4][4][4][100005];


bool isCompatible(const triplet& t1,  const triplet& t2){
	//check if triplets t1 and t2 are compatible
	if(t1.x==t2.x || t1.y==t2.y || t1.z==t2.z ){
		return 0;
	}
	return 1;
}

int Solution::solve(int n){
        fillTriplets();
	if(n<=0) return -1;
	
       //Bottom-up dp
	for(int i=1; i<=n; i++){
		for(int j=0; j<36; j++){
			if(i==1) dp[trip[j].x][trip[j].y][trip[j].z][i] = 1;
			else{
				ll temp = 0;
				
				for(int k=0; k<36; k++){
					if(isCompatible(trip[j], trip[k])){
						temp += dp[trip[k].x][trip[k].y][trip[k].z][i-1];
						temp %= mod;
					}
				}
				dp[trip[j].x][trip[j].y][trip[j].z][i] = temp;
			}
		}
	}
	
	ll ans = 0;
	for(int i=0; i<trip.size(); i++){
		ans = (ans + dp[trip[i].x][trip[i].y][trip[i].z][n])%mod;
	}
        int res = ans;

	return res;
}

https://leetcode.com/problems/tallest-billboard/
beutiful way of creating states


//when you have query, thinks its applications in offline way

// you can solve, normal state dp
https://www.interviewbit.com/problems/equal-average-partition/

// you can solve, O(n) in greedy and O(n*A[i]) in dp
https://www.interviewbit.com/problems/jump-game-array/

// you can solve, O(n*A[i]) in greedy and O(n) in dp
https://www.interviewbit.com/problems/min-jumps-array/

// you can solve, good cases involved
https://www.interviewbit.com/problems/max-product-subarray/

// you can solve
https://www.interviewbit.com/problems/regular-expression-match/
https://www.interviewbit.com/problems/regular-expression-ii/

// optimised dynamic programming, good cases 
https://leetcode.com/problems/make-array-strictly-increasing


ERRRRRRRRRRRRRRRRRROOORRRRRRRRRR:-

https://cses.fi/problemset/result/6236931/ book shop why memeoization fails kncpsack

http://www.usaco.org/index.php?page=viewproblem2&cpid=994  range dp why incorrect


