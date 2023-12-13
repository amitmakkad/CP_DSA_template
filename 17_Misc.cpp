// Always check:-
// Apply brute force (sometimes it passes in tight constraint problem)
// think greedy
// think binary search
// think dp
// think graph
// think seg tree

// Contents:-
// Meet in middle
// rectangle
// brute force runs 
// greedy wins
// greedy fails in knpasack
// PnC
// code median of median and quick select(both different)

Meet in the middle is a technique where the search space is divided into two
parts of about equal size. A separate search is performed for both of the parts,
and finally the results of the searches are combined using maps or set or arrays.
O(n2^n/2)

You are given an array of n numbers. In how many ways can you choose a subset of the numbers with sum x ?
vector<int> sol(vector<int>a){
    vector<int>m;
    int n=a.size();
    for(int i=0;i<(1<<n);i++){
        int sum=0;
        for(int j=0;j<n;j++){
            if((i&(1<<j))){
                sum+=a[j];
            }
        }
        m.push_back(sum);
    }
    sort(all(m));
    return m;
}
int main(){
    int n,k;
    cin>>n>>k;
    int a[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    vector<int>v;
    for(int i=0;i<n/2;i++){
        v.push_back(a[i]);
    }
    vector<int>x=sol(v);
    
    v.clear();
    for(int i=n/2;i<n;i++){
        v.push_back(a[i]);
    }
    
    vector<int>y=sol(v);
    
    int ans=0;
    for(auto i:x){
        int l=lower_bound(all(y),k-i)-x.begin();
        int r=upper_bound(all(y),k-i)-x.begin();
        ans+=r-l;
    }
    cout<<ans<<endl;
}



usaco bronze rectangle
(1) finding intersection area
(2) find no of corners coverd
(3) 4 cases coverd from top,left,bottom,right


before applying greedy think of brute force
like load balencing http://www.usaco.org/index.php?page=viewproblem2&cpid=617
cross roads2 http://www.usaco.org/index.php?page=viewproblem2&cpid=712
genomics https://codeforces.com/contest/863/problem/B
C. Jury Marks https://codeforces.com/contest/831/problem/C
when given final ans, construct intial ans
just find possible initial values then simulate and eveluate whether it comes or not 


sometimes in greedy
for(int i=0;i<n;i++){
    if(a[i]!=sorted_order(b[i])){
        ans++;
    }
}
works!!

https://usaco.guide/problems/usaco-785-out-of-place/solution
https://usaco.guide/problems/usaco-1227-photoshoot/solution


http://www.usaco.org/index.php?page=viewproblem2&cpid=509  All about Base

http://www.usaco.org/current/data/sol_whatbase_bronze.html    good way of applying b search



http://www.usaco.org/index.php?page=viewproblem2&cpid=642   feild reproduction
see how many case made: 4^4, you also have to think like that


https://usaco.guide/problems/usaco-923-painting-the-barn/solution   good Q 
2d prefix sum with good constructive mind


Q in which some order is given or queries order
think we go backwards or storing the data from queries and then intialization of required ds helps or not

Coordinate compression technique using array more better than map
sort(v.begin(), v.end());
v.resize(unique(v.begin(), v.end()) - v.begin());
vector<int> cnt(n);
for (auto &i : a) {
    int posl = lower_bound(cval.begin(), cval.end(), i.first) - cval.begin();
    int posr = lower_bound(cval.begin(), cval.end(), i.second + 1) - cval.begin();
    ++cnt[posl];
    --cnt[posr];
}


or use like
vpii v;
for(int i=0;i<n;i++){
    cin>>l>>r;
    v.push_back({l,1});
    v.push_back({r+1,-1});
}
sort(all(v));
int ans[n+1]={0};
for(int i=0;i<v.size()-1;i++){
    if(i>0){
        v[i].second=v[i].second+v[i-1].second;
    }
    ans[v[i].second]=ans[v[i].second]+v[i+1].first-v[i].first;
}

// median in used in choosing a value from array such that sum abs(a[i]-x) is minimized

// greedy fails in knapsack

// PnC

nCr+nCr-1=n+1Cr
nCr=n/r*n-1Cr-1
(1-x)^-n= 1+nx+n(n+1)/2x^2 ...
coffe of x_r = n+r-1Cr

Distribution of n chocklates to r kids = n+r-1Cr-1   (need r-1 sticks)

factorial everywhere in rank in repetion word
TTEERL rank??

EE
L
R
TT

    T               T           E          E     R      L
  ((4/2!)/2!)*5!  (4/2!)*4!   (0/2!)*3!  0*2!   1*1!    0! = 170
similar letters and confusion of cutting particular letter

selection of r items from n different items = nCr 
selection of r items from n identical items = 1

n items = p identical + q identical + s different

selection of r items = coffe of x_r in (x_0+x_1+...+x_p)(x_0+x_1+..+x_q)(x_0+x_1)^s 
arrangement of r items = coffe of x_r*r! in (x_0/0!+x_1/1!+...+x_p/p!)(x_0/0!+x_1/1!+..+x_q/q!)(x_0/0!+x_1/1!)^s

x1+x2+..xr=n

non negative integeral sol = n+r-1Cr-1
positive integeral sol = n-1Cr-1
positve interal sol , x1=x      x>=1
                      x2=x+y    y>=1
                      x3=x+y+z  z>=1  

Dearrangements D(n) number of ways in which each object noy go to its original place 
D(n)=n!(1/0!-1/1!+1/2!-1/3!...)
D(n)=(n-1)(D(n-1)+D(n-1))

Division and Distribution
total = m+n+n+p+p+p
(m+2n+3p)!*6!/m!n!n!2!p!p!p!3!

Distribution of r objects in n persons
coffe of r!*x_r in (1+x+x^2/2!+x^3/3!+..)^n 
                in e^(nx)

common errors:-
take input 
first logic, complexity, testcase then code
\n vs endl 
u-- v--
brackets in binary 
sometimes you need to finding only the last ans, no need of full simulation
nice code like for (int v=dest;v!=-1;v=parent[v])
while(-1) is true 
while(n){n=n/2;} not to reuse n
given n=1e5, but check all cases sometime it may possible you find n=1e3
dp[i]-dp[i-1]+mod%mod
int overflow
pq.insert(-val) for min heap or sorting
you can find firstsetbit using log also 
lsb vs msb      msb 1100111  lsb
check unordered_set<pii> works or not 
bipartite connection odd lenght cycle 
sometimes writting is better than thinking 
in lis dp, dp[i] gives lis end at i, not till i 
multiset vs set rember what to use, repeating elements 






To shuffle an array a of n elements (indices 0..n-1):
  for i from n - 1 downto 1 do
       j = random integer with 0 <= j <= i
       exchange a[j] and a[i]


