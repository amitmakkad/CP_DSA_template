// Contents:-
// common trick
// roman to int
// KMP
// z function
// rabbin krap 
// some q








if in some q, it is written you can make 'a' to 'o' and 'o' to 'a', do 'a'='#' and 'o'='#'







// common trick
long Solution::countSalutes(string A) {
    long salutes=0;
    long going_right=0;
    for(int i=0;i<A.size();i++){
        if(A[i]=='>'){
            going_right++;
        }
        else{
            salutes+=going_right;
        }
    }
    return salutes;
}

//roman to int
int Solution::romanToInt(string A) {
     string s=A;
     if (s.empty()) { return 0; }
        unordered_map<char, int> mp { {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000} };
        int sum = mp[s.back()];
        for (int i = s.size() - 2; i >= 0; --i) {
            sum += mp[s[i]] >= mp[s[i + 1]] ? mp[s[i]] : -mp[s[i]];
        }
        return sum;
}
//see we are going from back XIV and XVI

//KMP

//π[i] is the largest integer smaller than i such that P1 . . . Pπ[i] is a suffix of P1 . . . Pi
class Solution {
public:
    vector<int> prefix(string &s){
        int n=s.size();
        vector<int>lps(n,0);

        for(int i=1;i<n;i++){
            int j=lps[i-1];
            while(j>0&&s[i]!=s[j]){
                j=lps[j-1];
            }
            if(s[i]==s[j]){
                j++;
            }
            lps[i]=j;
        }
        return lps;
    }
    int strStr(string haystack, string needle) {
        vector<int>lps=prefix(needle);
        int j=0;
        for(int i=0;i<haystack.size();){
            if(haystack[i]==needle[j]){
                i++;
                j++;
            }
            else{
                if(j==0){
                    i++;
                }
                else{
                    j=lps[j-1];
                }
            }
            if(j==needle.size()){
                return i-needle.size();
            }
        }
        return -1;
    }

};
//You are given a string s. You can convert s to a palindrome by adding characters in front of it.
//Return the shortest palindrome you can find by performing this transformation.

//using 2pointer:-
class Solution {
public:
    string shortestPalindrome(string s) {
        int n = s.size();
        int i = 0;
        for (int j = n - 1; j >= 0; j--) {
            if (s[i] == s[j])
                i++;
        }
        if (i == n) {
            return s;
        }
        string remain_rev = s.substr(i, n);
        reverse(remain_rev.begin(), remain_rev.end());
        return remain_rev + shortestPalindrome(s.substr(0, i)) + s.substr(i);
    }
};

//kmp
class Solution {
public:
    string shortestPalindrome(string s) {
        int n=s.size();
        string g(s);
        reverse(g.begin(),g.end());
        s=s+"&"+g;
        int m=s.size();
        vector<int>lps(m,0);
        int len=0;
        for(int i=1;i<m;i++){
            if(s[i]==s[len]){
                len++;
                lps[i]=len;
            }
            else if(len==0){
                lps[i]=len;
            }
            else{
                len=lps[len-1];
                i--;
            }
        }
        len=n-lps[m-1];
        s=g.substr(0,len)+s.substr(0,n);
        return s;
        
    }
};

The Z-function for string s is an array of length n where the i-th element is equal to the 
greatest number of characters starting from the position i that coincide with the first characters of s.

z[i] is the length of the longest string from i that is, at the same time, a prefix of s and 
a prefix of the suffix of s starting at i.

z[i]=j  then s[0..j]=s[i..i+j]

z[0]=0
 a  a  a  b  a  a  b
[0, 2, 1, 0, 2, 1, 0]

// the [l, r) indices of the rightmost segment match

vector<int> z_function(string s){
    int n=s.size();
    vector<int>z(n);
    int l=0,r=0;
    for(int i=1;i<n;i++) {
        if(i<r){
            z[i]=min(r-i,z[i-l]);
        }
        while(i+z[i]<n&&s[z[i]]==s[i+z[i]]){
            z[i]++;
        }
        if(i+z[i]>r) {
            l=i;
            r=i+z[i];
        }
    }
    return z;
}



class Solution {
public:
    
    int strStr(string haystack, string needle) {
        string s=needle+'&'+haystack;
        int n=s.size();
        vector<int>z(n,0);
        int l=0,r=0;
        for(int i=1;i<n;i++){
            if(i<r){
                z[i]=min(z[i-l],r-i);
            }
            while(i+z[i]<n&&s[z[i]]==s[i+z[i]]){
                z[i]++;
            }
            if(i+z[i]>r){
                l=i;
                r=i+z[i];
            }
        }
        for(int i=0;i<n;i++){
            cout<<z[i]<<" ";
        }
        for(int i=0;i<n;i++){
            if(z[i]==needle.size()){
                return i-needle.size()-1;
            }
        }
        return -1;
    }

};

// rabin karp rolling polynimial hash function
class Solution {
public:
    
    int strStr(string haystack, string needle) {
        int n=haystack.size();
        int m=needle.size();

        if(m>n) return -1;
        long long p=31;
        long long hash_needle=0,hash_hay=0;
        long long mod=1e9+7;
        long long pwr=1;
        for(int i=0;i<m-1;i++){
            pwr=(pwr*p)%mod;
        }
        for(int i=0;i<m;i++){
            hash_needle=(hash_needle*p+needle[i]-'a')%mod;
            hash_hay=(hash_hay*p+haystack[i]-'a')%mod;
        }
        if(hash_needle==hash_hay){
            return 0;
        }
        for(int i=m;i<n;i++){
            hash_hay=(hash_hay-(haystack[i-m]-'a')*pwr)%mod;
            hash_hay=(hash_hay*p+(haystack[i]-'a'))%mod;
            if(hash_hay=hash_needle){
                if(haystack.substr(i-m+1,m)==needle)
                    return i-m+1;
            }
        }
        return -1;
    }

};

// String compression
// Given a string  
// $s$  of length  
// $n$ . Find its shortest "compressed" representation, that is: find a string  
// $t$  of shortest length such that  
// $s$  can be represented as a concatenation of one or more copies of  
// $t$ .
// A solution is: compute the Z-function of  
// $s$ , loop through all  
// $i$  such that  
// $i$  divides  
// $n$ . Stop at the first  
// $i$  such that  
// $i + z[i] = n$ . Then, the string  
// $s$  can be compressed to the length  
// $i$ .

// The proof for this fact is the same as the solution which uses the prefix function.

// mod everywhere
//https://cp-algorithms.com/string/z-function.html#search-the-substring applications
https://www.interviewbit.com/problems/minimum-characters-required-to-make-a-string-palindromic/
https://www.interviewbit.com/problems/minimum-appends-for-palindrome/         lps concept
https://www.interviewbit.com/problems/power-of-2/
https://www.interviewbit.com/problems/justified-text/
https://www.interviewbit.com/problems/pretty-json/
https://www.interviewbit.com/problems/convert-the-amount-in-number-to-words/


