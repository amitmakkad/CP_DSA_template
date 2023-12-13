// Contents:-
// max mutually disjoint sets 
// meetings rooms
// greedy q
// majourity element
// use bsearch to resist greedy
// job scheduling
// life guard time intersection
// some q















//Given a set of N intervals denoted by 2D array A of size N x 2, the task is to find the length of maximal set 
// of mutually disjoint intervals.
int Solution::solve(vector<vector<int> > &A) {
    int n=A.size();
    vector<pair<int,int>>v(n);
    for(int i=0;i<n;i++){
        v[i].first=max(A[i][0],A[i][1]);
        v[i].second=min(A[i][0],A[i][1]);
    }
    sort(v.begin(),v.end());
    int ans=1;
    int p=v[0].first;
    for(int i=1;i<n;i++){
        if(v[i].second>p){
            ans++;
            p=v[i].first;
        }
    }
    return ans;
}


// no of meeting rooms required, 2 methods
Use min heap to store the meeting rooms end time. If new meeting start time is greater or equal
than least element, update it.
If not, open a new meeting room. Report the min heap size at the end.

use map  m[start]++,m[end]--


//N children are standing in a line. Each child is assigned a rating value. You are giving candies to these 
// children subjected to the following requirements:
//Each child must have at least one candy. Children with a higher rating get more candies than their neighbors.
//What is the minimum number of candies you must give?

int Solution::candy(vector<int> &A) {
    int n=A.size();
    vector<int>a(n,1);
    for(int i=1;i<n;i++){
        if(A[i]>A[i-1]){
            a[i]=max(a[i],a[i-1]+1);
        }
    }
    for(int i=n-2;i>=0;i--){
        if(A[i]>A[i+1]){
            a[i]=max(a[i],a[i+1]+1);
        }
    }
    int sum=0;
    for(int i=0;i<n;i++){
        sum=sum+a[i];
    }
    return sum;
}

//majourity element
int Solution::majorityElement(const vector<int> &A) {
    int cnt=0;
    int element=A[0];
    for(int i=0;i<A.size();i++){
        if(element==A[i]){
            cnt++;
        }
        else{
            cnt--;
            if(cnt<0){
                element=A[i];
                cnt=0;
            }
        }
    }
    return element;
}

//use of median property in arrangement of min jumps such that group seating will happen
DIVIDE_GROUP - Editorial 
https://discuss.codechef.com/t/divide-group-editorial/ 
you just have to simulate, not actually do calculations
acuratly think of bouns in bserach whether 1e9 or 1e14
also you may have to use unsigned long long


//job schedule
#include<bits/stdc++.h>
using namespace std;

int main(){

    int n;                   // n be number of jobs
    cin>>n;

    vector<vector<int>>jobs(n,vector<int>(3,0));   //{profit,deadline,index}

    int max_deadline=0; 

    for(int i=0;i<n;i++){
        int deadline,profit;
        cin>>deadline>>profit;                       // taking input

        jobs[i][0]=profit;
        jobs[i][1]=deadline;
        jobs[i][2]=i+1;

        max_deadline=max(max_deadline,deadline);
    }

    sort(jobs.rbegin(),jobs.rend());                 // sorting jobs on basis of max profit
    
    // for(int i=0;i<n;i++){
    //     cout<<jobs[i][0]<<" "<<jobs[i][1]<<" "<<jobs[i][2]<<endl;
    
    // }


    int job_schedule[max_deadline]={0};      // job_schedule[i] store which job to be done in between [i,i+1] time 
    int total_profit=0;
    
    set<int,greater<int>>free_slots;
    
    for(int i=0;i<max_deadline;i++){
        free_slots.insert(i);
    }
    

    for(int i=0;i<n;i++){
        int deadline=jobs[i][1];
        int profit=jobs[i][0];                // first take highest profit job
        int index=jobs[i][2];

        int get_slot=-1;                       // binary search approach to find rightmost free slot before deadline 
        if(free_slots.size()==0||deadline<=*free_slots.rbegin()){
            continue;
        }
        else{
            get_slot=*free_slots.lower_bound(deadline-1);
            job_schedule[get_slot]=index;
            total_profit=total_profit+profit;
            free_slots.erase(get_slot);
        }
    }
    cout<<total_profit<<endl;
    for(int i=0;i<max_deadline;i++){
        cout<<i<<" "<<i+1<<" "<<job_schedule[i]<<endl;
    }


    return 0;
}









http://www.usaco.org/index.php?page=viewproblem2&cpid=784
#include<bits/stdc++.h>
using namespace std;
#define in int t;cin>>t;while(t--)
#define ya cout<<"YES"<<endl;
#define na cout<<"NO"<<endl;
#define ll long long
const long long mod=1e9+7;
#define de(x) cout<<"the debug element is "<<x<<endl;
#define all(x) x.begin(),x.end()
#define rep(i,a,b) for(int i=a;i<b;i++)
#define pb push_back
#define vi vector<int>
#define vvi vector<vector<int>>
//sort(v.rbegin(),v.rend());
//vector<int>::iterator it;
// bool comp(pair<int,int>&v1,pair<int,int>&v2){
//     if(v1.second==v2.second)
//     return v2.first>v1.first;
//     return v2.second>v1.second;             }
//*min_element(v.begin(), v.end());

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	freopen("lifeguards.in", "r", stdin);
	freopen("lifeguards.out", "w", stdout);
    int n;
    cin>>n;
    vector<pair<int,int>>v(n);
    for(int i=0;i<n;i++){
        cin>>v[i].first;
        cin>>v[i].second;
    }
    sort(all(v));
    if(n==1){
        cout<<0<<endl;
    }
    else if(n==2){
        cout<<max(v[0].second-v[0].first,v[1].second-v[1].first);
    }
    else{
        int s=0;
        s=s+v[0].second-v[0].first;
        int p=v[0].second;
        
        for(int i=1;i<n;i++){
            p=max(p,v[i-1].second);
            s=s+max(0,v[i].second-max(p,v[i].first));
        }
        //cout<<s<<endl;
        int ft[n]={0};
        if(v[0].second-v[1].first>0){
            ft[0]=v[0].second-v[1].first;
            
        }
        ft[0]=v[0].second-v[0].first-ft[0];
        int x=0;
        for(int i=0;i<n-1;i++){
            x=max(x,v[i].second);
        }
        if(x-v[n-1].first>0){
            ft[n-1]=x-v[n-1].first;
        }
        ft[n-1]=max(0,v[n-1].second-v[n-1].first-ft[n-1]);
     
        int y=v[0].second;
        for(int i=1;i<n-1;i++){
            y=max(y,v[i-1].second);
            ft[i]=ft[i]+max(0,y-v[i].first)+max(0,v[i].second-v[i+1].first);
            ft[i]=max(0,v[i].second-v[i].first-ft[i]);
        }
        
       
        int r=*min_element(ft,ft+n);
        cout<<s-r<<endl;
    }
    return 0;
}



