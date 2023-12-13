point update range sum

BIT == Segment tree == pbds

//range update point sum
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
    int n,q;
    cin>>n>>q;
	vector<int>bit(n+1,0);
	for(int i=0;i<n;i++){
	    int x;
        cin>>x;
        update_range(i+1,x,bit,n);
        update_range(i+2,-x,bit,n);
    }
    
    while(q--){
        int d;
        cin>>d;
        if(d==1){
            int a,b,u;
            cin>>a>>b>>u;
            update_range(a,u,bit,n);
            update_range(b+1,-u,bit,n);
        }
        else{
            int idx;
            cin>>idx;
            cout<<query_range(idx,bit,n)<<endl;
        }
    }

//https://cses.fi/problemset/task/1144/  Salary Queries
coordinate compression with values given in queries(big brain)
BIT implementation


//Distinct Values Queries https://cses.fi/problemset/task/1734 
tricky implementation


Tired of his stubborn cowlick, Farmer John decides to get a haircut. He has N
(1≤N≤105) strands of hair arranged in a line, and strand i is initially Ai micrometers long (0≤Ai≤N).
Ideally, he wants his hair to be monotonically increasing in length, so he defines the "badness" of his hair as the 
number of inversions: pairs (i,j) such that i<j and Ai>Aj
For each of j=0,1,…,N-1, FJ would like to know the badness of his hair if all strands with length greater than j
are decreased to length exactly j

int lowbit( int x ){
    return x&(-x);
}

ll query( int x ){
    ll res = 0;
    while( x ){
        res += tr[x];
        x -= lowbit(x);
    }
    return res;
}

void add( int x , int val ){
    while( x <= n+1 ){
        tr[x] += val;
        x += lowbit(x);
    }
}

int main( ) {
    cin >> n;
    for( int i = 1 ; i <= n ; i++ ){
        cin >> a[i];
        a[i]++;
        ans[a[i]] += query(n+1)-query(a[i]);
        add(a[i],1);
    }
    for( int i = 0 ; i < n ; i++ ){
        if( i ) ans[i] += ans[i-1];
        cout << ans[i] <<endl;
    }
    return 0;
}
//see diff in here and inv count (here we move from n-1 to 1)\

https://codeforces.com/blog/entry/11275  
pbds diff b/w less and less_equal
pbds A;  A.insert();  *A.find_by_order(0) ;  A.order_of_key(6);   *A.lower_bound(6); 
imp - A.erase(A.find_by_order(A.order_of_key(a[i])));
find_by_order() - returns an iterator to the k-th largest element (counting from zero)
order_of_key() - the number of items in a set that are strictly smaller than our item
think of less and less_equal(multiset, but erase not work)
However for searching, lower_bound and upper_bound work oppositely. (less_equal)
Also, lets say you want to erase x, use s.erase(s.upper_bound(x)) 

https://leetcode.com/contest/weekly-contest-349/problems/maximum-sum-queries/