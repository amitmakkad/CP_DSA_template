// Contents:-
// to do 
// gcd logic
// chess basics
// number to string 
// permutation rank without repeat 
// permutation rank with repeats 
// fibonacci
// A[i]-> A[A[i]]
// prime factors and divisors
// combination and dearrangements
// some q










// x+y=n gcd(x,y)
// y=n-x  -> gcd(x,n-x)=gcd(x,n)  *****************best logic 

//x can have O(log x) distinct prime factors
// __gcd(a,b) runs in log(ab)

//bishop,knight,etc in chess moves
Rook - elephant
bishop - camel 
knight - horse
pawn - army


// 1->A,28->AB, BAB=2+26*1+26*26*2
https://www.interviewbit.com/problems/excel-column-title/
string Solution::convertToTitle(int A) {
    string s="";
    while(A>0)
    {
        s=(char)('A'+(A-1)%26)+s;
        A=(A-1)/26;
    }return s;
}

// permutation rank with no repeats
https://www.interviewbit.com/problems/sorted-permutation-rank/
int Solution::findRank(string A) {
    vector<int>f(300,0);
    for(int i=0;i<A.size();i++){
        int x=A[i];
        f[x]++;
    }
    int ans=0;
    int n=A.size();
    for(int i=0;i<n-1;i++){
        int val=A[i];
        int above=0;
        for(int j=0;j<val;j++){
            if(f[j]>0){
                above+=f[j];
                
            }
        }
        above=(above*fact[n-i-1])%mod;
        ans=(ans+above)%mod;
        f[val]--;
    }
    return (ans+1)%mod;
}
//permutation rank with repeats
https://www.interviewbit.com/problems/sorted-permutation-rank-with-repeats/
int Solution::findRank(string A) {
    vector<int>f(300,0);
    for(int i=0;i<A.size();i++){
        int x=A[i];
        f[x]++;
    }

    long long int ans=0;
    int n=A.size();
    
    for(int i=0;i<n-1;i++){
        int val=A[i];
        long long int above=0;
        long long int p=1;
        for(int j=0;j<val;j++){
            if(f[j]>0){
                above+=f[j];
            }
        }
        for(int j=0;j<300;j++){
            p=(p*fact[f[j]])%mod;
        }
        above=(above*fact[n-i-1])%mod;
        long long int x=pwr(p,mod-2,mod);
        above=(above*x)%mod;
        ans=(ans+above)%mod;
        f[val]--;
    }
    return (ans+1)%mod;    
}

// nth fibonacci
long long mod=1e9+7;

void multiply(vector<vector<long long>>&F,vector<vector<long long>>M){
    int x =  (F[0][0]*M[0][0]%mod + F[0][1]*M[1][0]%mod)%mod;
    int y =  (F[0][0]*M[0][1]%mod + F[0][1]*M[1][1]%mod)%mod;
    int z =  (F[1][0]*M[0][0]%mod + F[1][1]*M[1][0]%mod)%mod;
    int w =  (F[1][0]*M[0][1]%mod + F[1][1]*M[1][1]%mod)%mod;
    
    F[0][0] = x;
    F[0][1] = y;
    F[1][0] = z;
    F[1][1] = w;
}
void pwr(vector<vector<long long>>&f,int n){
    if(n==0||n==1){
        return ;
    }
    vector<vector<long long>>m(2,vector<long long>(2,0));
    m[0][0]=1;
    m[0][1]=1;
    m[1][0]=1;
    
    pwr(f,n/2);
    multiply(f,f);
    if(n%2==1){
        multiply(f,m);
    }
}

int Solution::solve(int A) {
    int n=A;    // 1 1
                // 1 0
    vector<vector<long long>>f(2,vector<long long>(2,0)); 
    f[0][0]=1;
    f[0][1]=1;
    f[1][0]=1;
    n--;
    pwr(f,n);
    return f[0][0];
    
}

//Given an array A of size N. Rearrange the given array so that A[i] becomes A[A[i]] with O(1) extra space.
A = B + C * N
A % N = B
A / N = C

C = arr[arr[i]]
void Solution::arrange(vector<int> &A) {
    int i,n=A.size();
    for(i=0;i<n;i++) A[i]=A[i]+(A[A[i]]%n)*n;
    for(i=0;i<n;i++) A[i]/=n;
}

in some problems when you need to search for a value that is bias plus a multiple of some fixed value
instead of bruteforcing multiples from 1 to maxlimits
take everything in form od modulo of that fixed no 
https://codeforces.com/contest/1294/problem/D


// calculating prime factors
vector<int> factor(int n) {
	vector<int> ret;
	for (int i = 2; i * i <= n; i++) {
		while (n % i == 0) {
			ret.push_back(i);
			n /= i;
		}
	}
	if (n > 1) { ret.push_back(n); }
	return ret;
}
// or dividing by seive method
// how beautifully seive is used to calculate no of divisors of any no


// for getting all divisors 12->1,2,3,4,6,12;
for(int i=1;i*i<=x;i++){
    if(x%i==0){
        factors.push_back(i);
        if(i!=x/i) factors.push_back(i); 
    }
}




// nCk = n-1Ck + n-1Ck-1
// use dp to calculate

// The number of derangements of n numbers, expressed as !n, is the number of permutations 
// such that no element appears in its original position. Informally, it is the number of ways n
// hats can be returned to n people such that no person recieves their own hat.

// n! sum k=0 to n (-1)^k/(k!)
// D(n)=(n-1)*(D(n-1)+D(n-2))



https://cses.fi/problemset/result/6158164/ logic??
https://www.interviewbit.com/problems/city-tour/
https://www.interviewbit.com/problems/k-th-permutation/
https://www.interviewbit.com/problems/numbers-of-length-n-and-value-less-than-k/
https://www.interviewbit.com/problems/addition-without-summation/
https://www.interviewbit.com/problems/next-smallest-palindrome/
https://cses.fi/problemset/task/1082
https://cses.fi/problemset/result/6158424/  product analysis  (original no)^(count of div/2)
http://www.usaco.org/index.php?page=viewproblem2&cpid=862
https://codeforces.com/contest/1606/problem/E

