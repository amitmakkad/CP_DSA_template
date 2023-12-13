#include<bits/stdc++.h>
using namespace std;
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
typedef tree<int, null_type, less_equal<int>, rb_tree_tag, tree_order_statistics_node_update> pbds;
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
// pbds A;  A.insert();  *A.find_by_order(0) ;  A.order_of_key(6);   *A.lower_bound(6); A.erase(A.find_by_order(A.order_of_key(a[i])));

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    in{
        int n; 
        cin>>n;





    }
    return 0;
}


// Seive ............................................................................................
int N=1e7+1;
vector<int>is_prime(N+1,-1);
for(int i=2;i<=N;i++){
    if(is_prime[i]==-1&&i*i<=N){
        for(int j=i*i;j<=N;j=j+i){
            is_prime[j]=i;
        }
    }
}


// Factorial Power Inverse  ncr........................................................................
// (n-1)!mod n = n-1

const int N=1e6+5;

int pwr(int a,int expo,int mod){
	int r=1;
	while(expo>0){
		if(expo%2==1)
			r=a*r%mod;
		a=a*a%mod;
		expo=expo/2;
	}
	return r;
}

int fact[N],invfact[N];

void generate_factorial(){
	int p=mod;
	fact[0]=1;
	int i;
	for(i=1;i<N;i++){
		fact[i]=(i*fact[i-1])%p;
	}
	i--;
	invfact[i]=pwr(fact[i],p-2,p);
	for(i=i-1;i>=0;i--){
		invfact[i]=(invfact[i+1]*(i+1))%p;
	}
}
void combination(){
    int N=1e3+5;
    c.resize(N+1,vector<int>(N+1,0));
    for(int i=0;i<=N;i++){
        c[i][0]=1;
        c[i][i]=1;
    }
    for(int i=1;i<=N;i++){
        for(int j=1;j<i;j++){
            c[i][j]=(c[i-1][j-1]+c[i-1][j])%mod;
        }
    }
}

int ncr(int n,int r){
	if(r > n || n < 0 || r < 0)
    return 0;
	
    return (((fact[n]*invfact[r])%mod)*invfact[n-r])%mod;
}
int tot_phi(int n){
    int result=n;
    for(int p=2;p*p<=n;p++){
        if(n%p==0){
            while(n%p==0){
                n=n/p;
            }
            result-=result/p;
        }
    }
    if(n>1){
        result-=result/n;
    }
    return result;
}

//Disjoint set union...................................................................................


int find(int *parent,int *rank,int x){
    if(parent[x]==x){
        return x;
    }
    parent[x]=find(parent,rank,parent[x]);
    return parent[x];
}
void unions(int *parent,int *rank,int x,int y){
    int x_rep=find(parent,rank,x);
    int y_rep=find(parent,rank,y);
    
    if(x_rep==y_rep)
        return;
    
    if(rank[x_rep]<rank[y_rep]){
        parent[x_rep]=y_rep;
    }
    else if(rank[x_rep]>rank[y_rep]){
        parent[y_rep]=x_rep;
    }
    else{
        parent[y_rep]=x_rep;
        rank[x_rep]++;
    }
}

signed main(){
    
    int parent[n];
    int rank[n];
    for(int i=0;i<n;i++){
        parent[i]=i;
        rank[i]=0;
    }
    
    return 0;
}


// segment tree...................................................................................

// tree arg 1 indexed
// a arg 0 indexed
void build(int *a,int *tree,int node, int st,int end){
    if(st==end){
        tree[node]=a[st];
        return;
    }
    int mid=(st+end)/2;
    build(a,tree,2*node,st,mid);
    build(a,tree,2*node+1,mid+1,end);
    tree[node]=tree[2*node]+tree[2*node+1];
}
void build_max(int *a, int *tree,int node,int st,int end){
    if(st==end){
        tree[node]=a[st];
        return;
    }
    int mid=(st+end)/2;
    build_max(a,tree,2*node,st,mid);
    build_max(a,tree,2*node+1,mid+1,end);
    tree[node]=max(tree[2*node],tree[2*node+1]);
}
int query(int *a,int *tree,int node,int st,int end,int l,int r){
    if(st>r||end<l){
        return 0;
    }
    if(l<=st&&end<=r){
        return tree[node];
    }
    int mid=(st+end)/2;
    int q1=query(a,tree,2*node,st,mid,l,r);
    int q2=query(a,tree,2*node+1,mid+1,end,l,r);
    return q1+q2;
}
int query_max(int *a,int *tree,int node,int st,int end,int l,int r){
    if(st>r||end<l)
    return INT_MIN;
    if(l<=st&&end<=r)
    return tree[node];
    
    int mid=(st+end)/2;
    int q1=query_max(a,tree,2*node,st,mid,l,r);
    int q2=query_max(a,tree,2*node+1,mid+1,end,l,r);
    return max(q1,q2);
}
void update(int *a,int *tree,int node,int st,int end,int idx,int value){
    if(st==end){
        a[st]=value;
        tree[node]=value;
        return ;
    }
    int mid=(st+end)/2;
    if(idx<=mid){
        update(a,tree,2*node,st,mid,idx,value);
    }
    else{
        update(a,tree,2*node+1,mid+1,end,idx,value);
    }
    tree[node]=max(tree[2*node],tree[2*node+1]);
    
}
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
	int n;
	cin>>n;
	
    int a[n];
    int tree[4*n];
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    build_max(a,tree,1,0,n-1);
    for(int i=0;i<5;i++){
        cout<<tree[i+1]<<" ";
    }
    update(a,tree,1,0,n-1,3,7);
    cout<<query_max(a,tree,1,0,n-1,1,3);
    
    return 0;
}

// fenwick tree..............................point update range sum...........................................

//1 indexed
int query(int index,vector<int>&bit){
    int ans=0;
    while(index>0){
        ans=ans+bit[index];
        index=index-(index&(-index));
    }
    return ans;
}

int sum_q(int l,int r,vector<int>&bit){
    return query(r,bit)-query(l-1,bit);
}

void update(int index,int val,vector<int>&bit,int n){
    while(index<=n){
        bit[index]=bit[index]+val;
        index=index+(index&(-index));
    }
}


signed main(){
    
    int n;
	cin>>n;
	vector<int>bit(n+1,0);
	vector<int>a(n+1,0);

	for(int i=1;i<=n;i++){
	    cin>>a[i];
	    update(i,a[i],bit,n);
	}
	int l=1,r=n;
	cout<<sum_q(l,r,bit)<<endl;

    return 0;
}

// range update point sum
// A[L,....] +increment     so terms above l have + increment
// A[R+1,...] -increment   so after r+1 net=0 
int query_range(int index,vector<int>&bit,int n){
    int ans=0;
    while(index>0){
        ans=ans+bit[index];
        index=index-(index&(-index));
    }
    return ans;
}
void update_range(int index,int val,vector<int>&bit,int n){
    while(index<=n){
        bit[index]=bit[index]+val;
        index=index+(index&(-index));
    }
}

signed main(){
    int n;
	cin>>n;
    vector<int>bit(n+1,0);
	for(int i=0;i<n;i++){
	    int x;
        cin>>x;
        update_range(i+1,x,bit,n);
        update_range(i+2,-x,bit,n);
    }
	int l=1,r=n;
	int val;
	update_range(l,val,bit,n);
	update_range(r+1,-val,bit,n);
	
	int index=0;
	index++;            // 1 based indexing
	cout<<query_range(index,bit,n)<<endl;

    return 0;
}

// count inversion using fenwick tree

int get_sum(int index,vector<int>&bit){
    int sum=0;
    while(index>0){
        sum=sum+bit[index];
        index=index-(index&(-index));
    }
    return sum;
}
void update_inv(int index,vector<int>&bit,int n){
    while(index<=n){
        bit[index]++;
        index=index+(index&(-index));
    }
    
}
signed main(){
    int n;
	cin>>n;
    vector<int>a(n);
    for(int i=0;i<n;i++){
	    cin>>a[i];
	}
    vector<int>temp=a;
    sort(temp.begin(),temp.end());
    for(int i=0;i<n;i++){
        a[i]=lower_bound(temp.begin(),temp.end(),a[i])-temp.begin()+1;
    }
    
    vector<int>bit(n+1,0);
    int inv=0;
    for(int i=n-1;i>=0;i--){
        inv=inv+get_sum(a[i]-1,bit);
        update_inv(a[i],bit,n);
    }
    
    cout<<inv<<endl;

    return 0;
}





// kickstart template
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


signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int g;
    cin>>g;
    for(int t=1;t<=g;t++){
        int n; 
        cin>>n;
        int ans=0;
        
        
        
        
        
        cout<<"CASE #"<<t<<": "<<ans<<endl;
    }
    return 0;
}





// old template.......................................................................................

#include<bits/stdc++.h>
using namespace std;
//*min_element(v.begin(), v.end());
//vector<vector<int>>v(board_size,vector<int>(board_size,0));
#define in int t;cin>>t;while(t--)
#define ya cout<<"YES"<<endl;
#define na cout<<"NO"<<endl;
#define f(a,b) for(long long int i=a;i<b;i++) 
#define ll long long
const long long int mod=1e9+7;
#define de(x) cout<<"the debug element is "<<x<<endl;
#define vec vector<int>v
#define all(x) x.begin(),x.end()
//if (int x = f(); is_good(x)) {use_somehow(x);}
//cout << bitset<20>(n) << "\n";
//if (binary_search(all(vec), key))
//sort(v.rbegin(),v.rend());
//vector<int>::iterator it;
// sort(all(vec));
// vec.resize(unique(all(vec)) - vec.begin());

// int n = nxt();
// vector<int> a(n);
// generate(all(a), nxt);
// bool comp(pair<int,int>&v1,pair<int,int>&v2){
//     if(v1.second==v2.second)
//     return v2.first>v1.first;
//     return v2.second>v1.second;             }
// int sum=0;
// int s=accumulate(a,a+4,sum);
// int b[4];
// partial_sum(a,a+4,b);
// for(int i=0;i<4;i++){
//     cout<<b[i]<<" ";
// }

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    in{
        int n; 
        cin>>n;





    }
    return 0;
}




  
    
