// Contents:-
// some facts
// heap sort 
// simple qs
// lru using queue and map(not good approach)
// minimum rooms,etc based on depature time 
// kth largest elements
// some questions


faster hash with less collosion - https://usaco.guide/gold/hashmaps?lang=cpp
remember in cf, un_ordered set get hacked easily


for 2d coordinates use multiset<pii>st1,st2;rep(i,0,n-1){
    st1.insert({v[i].F,v[i].S});
    st2.insert({v[i].S,v[i].F});
}

for (auto &x : m) {
	x.second = 1234;  // Change all values to 1234
}imp of &

maintain a list of all the keys you want to erase and erase them after the iteration finishes.


// full BT has 0 or 2 childs
// complete BT has all levels except last has max possible number of nodes
// heap is complete BT


// heap sort
#include<bits/stdc++.h>
using namespace std;

// sorts A[i],A[left],A[right] and recusively call for its child if not satisfies
void heapify(int *a,int i,int n){   //O(logn)
    int left=2*i;
    int right=2*i+1;

    int max_index=i;
    if(a[i]<a[left]&&left<=n){
        swap(a[i],a[left]);
        max_index=left;
    }
    if(a[i]<a[right]&&right<=n){
        swap(a[i],a[right]);
        max_index=right;
    }
    if(max_index!=i){
        heapify(a,max_index,n);
    }

}

void heap_sort(int *a,int n){
    for(int i=n/2;i>=1;i--){
        heapify(a,i,n);             //build heap looks O(nlogn) but its O(n)
    }
    int size=n;
    for(int i=1;i<=n;i++){
        swap(a[1],a[size]);
        size--;
        heapify(a,1,size);
    }
}
int main(array a){  //1 indexed array size=n+1
    heap_sort(a,n);
    return 0;
}



//Given an integer array A of size N containing 0's and 1's only.
//You need to find the length of the longest subarray having count of 1’s one more than count of 0’s

int Solution::solve(vector<int> &A) {
    int n=A.size();
    int sum=0;
    int ans=0;
    map<int,int>m;
    for(int i=0;i<n;i++){
        if(A[i]==0){
            A[i]--;
        }
        sum+=A[i];
        if(sum==1){
            ans=i+1;
        }
        else if(m.find(sum)==m.end()){
            m[sum]=i;
        }
        if(m.find(sum-1)!=m.end()){
            ans=max(ans,i-m[sum-1]);
        }
    }
    return ans;
}
//Given two equally sized 1-D arrays A, B containing N integers each.
//A sum combination is made by adding one element from array A and another element of array B.
//Return the maximum C valid sum combinations from all the possible sum combinations.

vector<int> Solution::solve(vector<int> &A, vector<int> &B, int C) {
    sort(A.begin(),A.end());
    sort(B.begin(),B.end());
    priority_queue<int,vector<int>,greater<int>>pq;
    for(int i=A.size()-1;i>=0;i--){
        for(int j=B.size()-1;j>=0;j--){
            int x=A[i]+B[j];
            if(pq.size()<C){
                pq.push(x);
            }
            else{
                int mini=pq.top();
                if(x>mini){
                    pq.pop();
                    pq.push(x);
                }
                else{
                    j=-1;
                }
            }
        }
    }
    vector<int>ans;
    while(pq.size()>0){
        ans.push_back(pq.top());
        pq.pop();
    }
    reverse(ans.begin(),ans.end());
    return ans;
}

//lru using queue and map
deque<int>q;
map<int,int>m;
int c=0;

LRUCache::LRUCache(int capacity) {
    q.clear();
    m.clear();
    c=capacity;
}

int LRUCache::get(int key) {
    if(m.find(key)==m.end()){
        return -1;
    }
    else{
        vector<int>v;
        for(auto it=q.begin();it!=q.end();it++){
            if(*it!=key){
                v.push_back(*it);
            }
        }
        q.clear();
        for(int i=0;i<v.size();i++){
            q.push_back(v[i]);
        }
        q.push_back(key);
        return m[key];
    }
}
void LRUCache::set(int key, int value) {
    if(m.find(key)==m.end()){
        if(q.size()>=c){
            int old_key=q.front();
            q.pop_front();
            m.erase(m.find(old_key));
            m[key]=value;
            q.push_back(key);
        }
        else{
            q.push_back(key);
            m[key]=value;
        }
    }
    else{
        m[key]=value;
        vector<int>v;
        for(auto it=q.begin();it!=q.end();it++){
            if(*it!=key){
                v.push_back(*it);
            }
        }
        q.clear();
        for(int i=0;i<v.size();i++){
            q.push_back(v[i]);
        }
        q.push_back(key);
    }    
}


// In a movie festival, n movies will be shown. Syrjäläs movie club consists of k members, 
// who will be all attending the festival.
// You know the starting and ending time of each movie. What is the maximum total number of 
// movies the club members can watch entirely if they act optimally?

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
using namespace std;

int main() {
	int n, k;
	cin >> n >> k;
	vector<pair<int, int>> v(n);
	for (int i = 0; i < n; i++)  // read start time, end time
		cin >> v[i].second >> v[i].first;
	sort(begin(v), end(v));  // sort by end time

	int maxMovies = 0;
	multiset<int> end_times;  // times when members will finish watching movies
	for (int i = 0; i < k; ++i) end_times.insert(0);

	for (int i = 0; i < n; i++) {
		auto it = end_times.upper_bound(v[i].second);
		if (it == begin(end_times)) continue;
		// assign movie to be watched by member in multiset who finishes at time
		// *prev(it)
		end_times.erase(--it);
		// member now finishes watching at time v[i].first
		end_times.insert(v[i].first);
		++maxMovies;
	}

	cout << maxMovies;
}







https://cses.fi/problemset/task/1164

There is a large hotel, and n customers will arrive soon. Each customer wants to have a single room.
You know each customers arrival and departure day. Two customers can stay in the same room
if the departure day of the first customer is earlier than the arrival day of the second customer.
What is the minimum number of rooms that are needed to accommodate all customers? And how can the rooms be allocated?

#include <algorithm>
#include <iostream>
#include <queue>
using namespace std;

const int MAX_N = 2e5;

int N;
int ans[MAX_N];
vector<pair<pair<int, int>, int>> v(MAX_N);

int main() {
	cin >> N;
	v.resize(N);
	for (int i = 0; i < N; i++) {
		cin >> v[i].first.first >> v[i].first.second;
		v[i].second = i;  // store the original index
	}
	sort(v.begin(), v.end());

	int rooms = 0, last_room = 0;
	priority_queue<pair<int, int>> pq;  // min heap to store departure times.
	for (int i = 0; i < N; i++) {
		if (pq.empty()) {
			last_room++;
			// make the departure time negative so that we create a min heap
			// (cleanest way to do a min heap... default is max in c++)
			pq.push(make_pair(-v[i].first.second, last_room));
			ans[v[i].second] = last_room;
		} else {
			// accessing the minimum departure time
			pair<int, int> minimum = pq.top();
			if (-minimum.first < v[i].first.first) {
				pq.pop();
				pq.push(make_pair(-v[i].first.second, minimum.second));
				ans[v[i].second] = minimum.second;
			}

			else {
				last_room++;
				pq.push(make_pair(-v[i].first.second, last_room));
				ans[v[i].second] = last_room;
			}
		}

		rooms = max(rooms, int(pq.size()));
	}

	cout << rooms << "\n";
	for (int i = 0; i < N; i++) { cout << ans[i] << " "; }
}
//meetings rooms
https://www.interviewbit.com/problems/meeting-rooms/
Solution 1: Use min heap to store the meeting rooms end time. If new meeting start time is greater or equal than least element, update it.
If not, open a new meeting room. Report the min heap size at the end.
Time Complexity : O(NlogN).

Solution 2: Two sorted array of start time and end time. Two pointers to iterator start array and end array. Iterate the time line, the current time active meeting is num of start minus num of end.
Since need sort, still O(nlogn) solution, but fast than solution 1.

solution 3: use map, m[start]++, m[end]--

https://leetcode.com/problems/meeting-rooms-iii/
earlier you made mistake in time shifting, end+new_meeting.end-new_meeting.start 
class Solution {
public:
    int mostBooked(int k, vector<vector<int>>& meetings) {
        set<int>rooms;
        for(int i=0;i<k;i++){
            rooms.insert(i);
        }
        
        
        priority_queue<pair<long long,int>,vector<pair<long long,int>>,greater<pair<long long,int>>>q;

        sort(meetings.begin(),meetings.end());

        vector<int>freq(k,0);
        for(int i=0;i<meetings.size();i++){
            long long start=meetings[i][0];
            while(q.size()>0){
                pair<long long,int> p=q.top();
                long long end=p.first;
                int room=p.second;
                if(end<=start){
                    rooms.insert(room);
                    q.pop();
                }
                else{
                    break;
                }
            }
            if(rooms.size()==0){
                pair<long long,int> p=q.top();
                long long end=p.first;
                int room=p.second;
                q.pop();
                freq[room]++;
                q.push({end+meetings[i][1]-meetings[i][0],room});
            }
            else{
                int room=*rooms.begin();
                rooms.erase(rooms.begin());
                freq[room]++;
                q.push({meetings[i][1],room});
            }
        }

        int x=*max_element(freq.begin(),freq.end());
        for(int i=0;i<k;i++){
            if(x==freq[i]){
                return i;
            }
        }
        return 0;
    }
};

Kth Largest Element in an Array
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int,vector<int>,greater<int>>pq;
        int n=nums.size();
        
        for(int i=0;i<n;i++){
            pq.push(nums[i]);
            if(pq.size()>k){
                pq.pop();
            }
        }
        return pq.top();
    }
};
// sorting
// minheap
// counting sort
// quicksort partition algo




https://www.interviewbit.com/problems/merge-k-sorted-lists/
https://www.interviewbit.com/problems/4-sum/
https://www.interviewbit.com/problems/ways-to-form-max-heap/
http://www.usaco.org/index.php?page=viewproblem2&cpid=1158 
code below:-
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
    
    int k,m,n;
    cin>>k>>m>>n;
    int pos[k],t[k];
    for(int i=0;i<k;i++){
        cin>>pos[i]>>t[i];
        
    }
    vector<int>f(m,0);
    for(int i=0;i<m;i++){
        cin>>f[i];
        
    }
    
    int diff[k];
    for(int i=0;i<k;i++){
        auto it=upper_bound(f.begin(),f.end(),pos[i]);
        
        if(it==f.end()){
            it--;
            diff[i]=abs(*it-pos[i]);
        }
        else{
            diff[i]=abs(*it-pos[i]);
            if(it!=f.begin()){
                it--;
                diff[i]=min(diff[i],abs(*it-pos[i]));
            }
        }
        
    }
    
    map<float,int>ms;
    
    for(int i=0;i<k;i++){
        float a=pos[i]-diff[i];
        a+=0.5;
        float b=pos[i]+diff[i];
        b-=0.5;
        ms[a]+=t[i];
        ms[b]-=t[i];
    }
    int sum=0;
    set<int>s;
    for(auto x:ms){
        
        sum+=x.second;
        cout<<x.second<<" "<<sum<<endl;
        s.insert(sum);
    }
    
    int ans=0;
    while(n--&&s.size()>0){
        auto it=s.end();
        it--;
        cout<<*it<<endl;
        ans+=*it;
        s.erase(it);
    }
    cout<<ans<<endl;
    return 0;
}



