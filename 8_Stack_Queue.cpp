// Contents:-
// some facts
// stack basic q 
// nxt smaller
// rain water trap 
// max area in rectangle
// 2nd definition of kadane
// sliding median
// stack gfg 
// queue number using 5,6
// some q


// q.back()  also works

//ans=max(ans,s%mod); is wring
//ans=max(ans,s);
//return ans%mod

//for(int i=n-1;i>=0;i--) ans.push_back(f(i))
//remember you have to do reverse of ans

//errors- use multiset not set, use of ans=max(ans,v[i]) till i<b 


//Given a string A denoting an expression. It contains the following operators '+', '-', '*', '/'. 
// Chech whether A has redundant braces or not.

int Solution::braces(string A) {
    vector<char>v;
    for(int i=0;i<A.size();i++){
        if(A[i]=='('||A[i]=='+'||A[i]=='-'||A[i]=='*'||A[i]=='/'){
            v.push_back(A[i]);
        }
        else if(A[i]==')'){
            if(v.back()=='('){
                return 1;
            }
            else{
                while(v.size()>0&&v.back()!='('){
                    v.pop_back();
                }
                v.pop_back();
            }
        }
    }
    return 0;    
}

//nxt smaller
vector<int> Solution::prevSmaller(vector<int> &A) {
    // remember to reverse in scheme while traversing i from n-1 to 0
    // check inequality sign carefully
    vector<int>ans;
    stack<int>s;
    for(int i=0;i<A.size();i++){
        while(s.size()>0&&s.top()>=A[i]){
            s.pop();
        }  
        if(s.size()==0){
            ans.push_back(-1);
        }
        else{
            ans.push_back(s.top());
        }
        s.push(A[i]);
    }
    return ans;
}




// rain water trap problem
int Solution::trap(const vector<int> &A) {
    int n=A.size();
    int left[n],right[n];
    int ans=0;
    left[0]=A[0];
    for(int i=1;i<n;i++){
        left[i]=max(left[i-1],A[i]);
    }
    right[n-1]=A[n-1];
    for(int i=n-2;i>=0;i--){
        right[i]=max(right[i+1],A[i]);
    }
    for(int i=1;i<n-1;i++){
        int var=min(left[i-1],right[i+1]);
        if(var>A[i]){
            ans+=var-A[i];
        }
    }
    return ans;
}




//max area in matrix
#include<bits/stdc++.h>
using namespace std;
#define ya cout<<"YES"<<endl;
#define na cout<<"NO"<<endl;
const long long mod=1e9+7;
#define int long long   // remember during bits operation
#define in int t;cin>>t;while(t--)
#define de(x) cout<<"the debug element is "<<x<<endl;
#define all(x) x.begin(),x.end()
#define rep(i,a,b) for(int i=a;i<b;i++)
#define endl "\n"          // remember to change in interactive problem
#define vi vector<int> 
#define vvi vector<vector<int>> 
#define pii pair<int,int> 
#define vpii vector<pair<int,int>> 
#define vvpii vector<vector<pair<int,int>>> 
#define pb push_back;
#define ff first
#define ss second
//sort(v.rbegin(),v.rend());
// bool comp(pair<int,int>&v1,pair<int,int>&v2){
//     if(v1.second==v2.second)
//     return v2.first>v1.first;
//     return v2.second>v1.second;             }

int max_area(vector<int>&a){
    int n=a.size();
    vector<int>prev_s;
	vector<int>next_s;
	stack<int>s;
	
	for(int i=0;i<n;i++){
	    while(s.size()>0&&a[s.top()]>=a[i]){
	        s.pop();
	    }
	    if(s.size()==0){
	        prev_s.push_back(-1);
	    }
	    else{
	        prev_s.push_back(s.top());
	    }

	    s.push(i);
	}

	while(s.size()>0) s.pop();
	for(int i=n-1;i>=0;i--){
	    while(s.size()>0&&a[s.top()]>=a[i]){
	        s.pop();
	    }
	    if(s.size()==0){
	        next_s.push_back(n);
	    }
	    else{
	        next_s.push_back(s.top());
	    }
	   
	    s.push(i);
	}

	reverse(all(next_s));
	int ans=0;
	for(int i=0;i<n;i++){
	    ans=max(ans,a[i]*(next_s[i]-prev_s[i]-1));
	}
	return ans;
}



signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
	int n,m;
	cin>>n>>m;
	vector<vector<int>>a(n,vector<int>(m,0));
	for(int i=0;i<n;i++){
	    for(int j=0;j<m;j++){
	        char c;
	        cin>>c;
	        if(c=='.') a[i][j]=1;
	    }
	}
	for(int j=0;j<m;j++){
	    for(int i=1;i<n;i++){
	        if(a[i][j]==1){
	            a[i][j]+=a[i-1][j];
	        }
	    }
	}
	int ans=0;
	for(int i=0;i<n;i++){
	    ans=max(ans,max_area(a[i]));
	}
	cout<<ans<<endl;    
    return 0;
}

//Given an array of n integers, your task is to find the maximum sum of values in a contiguous subarray
//  with length between a and b

// 2nd definition of kadane algorithm 
sum=max(prefix[i]-min 1<j<i(prefix[j]))
    
int main(){
    int n,a,b;
	cin>>n>>a>>b;
	vector<int>v(n,0);
	for(int i=0;i<n;i++){
	    cin>>v[i];
	}
	for(int i=1;i<n;i++){
	    v[i]+=v[i-1];
	}
	multiset<int>s;
	int ans=v[a-1];
	
	for(int i=a;i<n;i++){
	    s.insert(v[i-a]);
	    if(i-b-1>=0){
	        s.erase(s.find(v[i-b-1]));
	    }
	    ans=max(ans,v[i]-*s.begin());
	    if(i<b){
	        ans=max(ans,v[i]);
	    }
	    
	}
	cout<<ans<<endl;
}
//errors- use multiset not set, use of ans=max(ans,v[i]) till i<b 

//sliding median
using pbds
using fenwick tree
using 2 multisets  https://usaco.guide/problems/cses-1076-sliding-median/solution

// stack in gfg

// k stacks in array 

// getmin in O(1) space handling negative also
finding getmin of stack

i using auxillary stack

ii using O(1) space
struct MyStack {
 
    stack<int> s;
    int min;
 
void push(int x) {
 
      if(s.empty() ) {
         min=x;
         s.push(x);
      }
      else if(x<=min){
          s.push(x-min);
          min=x;
      }else{
      s.push(x);
      }
   }
 
int pop() {

    int t=s.top();s.pop();
    if(t<=0){
        int res=min;
        min=min-t;
        return res;
    }else{
        return t;
    }
   }

int top() {
    int t=s.top();
    return ((t<=0)? min : t);
   }
 
int getMin() {
      return min;
   }
};
// max_area in 1 traversal
int getMaxArea(int arr[],int n){
    stack <int> s;
    int res=0;
    int tp;
    int curr;
    
    for(int i=0;i<n;i++){
        while(s.empty()==false && arr[s.top()]>=arr[i]){
            tp=s.top();s.pop();
            curr=arr[tp]* (s.empty() ? i : i - s.top() - 1);
            res=max(res,curr);
        }
        s.push(i);
    }
    while(s.empty()==false){
        tp=s.top();s.pop();
        curr=arr[tp]* (s.empty() ? n : n - s.top() - 1);
        res=max(res,curr);
    }
    
    return res;
    
}


// Given a number n, print first n number(in increasing order) such that all these numbers have digits in set {5, 6}

void printFirstN(int n) 
{ 
    queue<string> q;
    
    q.push("5");
    q.push("6");
    
    for(int i = 0; i < n; i++)
    {
        string curr = q.front();
       
        cout << curr << " ";
        
        q.pop();
        
        q.push(curr + "5");
        q.push(curr + "6");
    }
        
} 


https://codeforces.com/gym/101102/problem/D
https://cses.fi/problemset/task/2416
https://usaco.guide/problems/usaco-923-painting-the-barn/solution  2d prefix sum and kadane
https://leetcode.com/problems/daily-temperatures/
https://leetcode.com/problems/shortest-unsorted-continuous-subarray/
https://leetcode.com/problems/next-greater-element-i/
https://leetcode.com/problems/next-greater-element-ii/
https://leetcode.com/problems/online-stock-span/
https://leetcode.com/problems/132-pattern/
https://leetcode.com/problems/sum-of-subarray-minimums/