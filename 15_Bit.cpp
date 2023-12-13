//whenever power of 2 is there bitmask
//think meet in the middle

//Different Bits Sum Pairwise
int Solution::cntBits(vector<int> &A) {
    long long int ans=0;
    long long int mod=1e9+7;
    long long int n=A.size();
    
    
    for(int i=0;i<31;i++){
        long long int c=0;
        for(int j=0;j<A.size();j++){
            if((A[j]&(1<<i))){
                c++;
            }
        }
        c=(c*(n-c))%mod;
        c=(c*2)%mod;
        ans=(ans+c)%mod;
    }
    return ans;
}
//Given an array of integers, every element appears thrice except for one, which occurs once. Find that element that does not appear thrice.
int Solution::singleNumber(const vector<int> &A) {
    int a[32]={0};
    int b[32]={0};
    

    for(int i=0;i<32;i++){
        for(int j=0;j<A.size();j++){
            if((A[j]&(1<<i)))
            a[i]++;
        }
        a[i]=a[i]%3;
    }
    int ans=0;
    for(int i=0;i<32;i++){
        if(a[i]==1){
            ans=(ans|(1<<i));
        }
    }

    return ans;

}

//Given an integer array A of N integers, find the pair of integers in the array which have minimum XOR value. Report the minimum XOR value.
int Solution::findMinXor(vector<int> &A) {
    int ans=INT_MAX;
    sort(A.begin(),A.end());
    for(int i=0;i<A.size()-1;i++){
        ans=min(ans,A[i]^A[i+1]);
    }
    return ans;
}

//Given a positive integer A, the task is to count the total number of set bits in the 
// binary representation of all the numbers from 1 to A.
int Solution::solve(int A) {
   long long  int ans=0;
   
   int bits=log10(A)/log10(2);
   //cout<<bits;
   for(int i=0;i<=bits;i++)
   {
       int c=pow(2,i),cnt=0;
       int set=(A+1)/c;
       cnt=(set/2)*c;
       int incomp=(A+1)%c;
       if(set%2!=0) cnt+=incomp;
       
       ans+=cnt;
   }
    return ans%1000000007;
}



https://www.interviewbit.com/problems/divide-integers/
https://www.interviewbit.com/problems/palindromic-binary-representation/
https://www.codechef.com/problems/MINORPATH  //checking possibilities, then adding bit by bit ans
https://codeforces.com/contest/1466/problem/E



class Solution {
public:
//by dp and memorization
//tc=o(n*n) n=size of words
//sc=o(n*n)+o(n) n=stack space 
int dp[1001][1001];
   
    bool cheak(string &s1,string &s2)
    {
        if(s1.size()==s2.size())  return false;
        if(s1.size()>s2.size()+1) return false;
         int i=0;
         int j=0;
        while(i<s1.size())
            if(j<s2.size() && s1[i]==s2[j])
                i++,j++;
            else
                i++;
        if(i==s1.size() && j==s2.size())
            return true;
        else
            return false;
    }
    int find(int i,int pre,vector<string>& words){
        if(i==-1){
            return 0;
        }
        if(dp[i][pre+1]!=-1){
            return dp[i][pre+1];
        }
        //take
        int take=0,not_take=0;
        if(pre==-1 || cheak(words[pre],words[i])==true){
            take=1+find(i-1,i,words);
        }
        //not take
        not_take=find(i-1,pre,words);
        return dp[i][pre+1]=max(take,not_take);
    }
    int longestStrChain(vector<string>& words) {
        int n=words.size();
        memset(dp,-1,sizeof(dp));
        sort(words.begin(),words.end(),com);
        return find(n-1,-1,words);
    }
    private:
    static bool com(string& a,string& b){
        return a.size()<b.size();
    }
};