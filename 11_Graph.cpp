Contents:-
basics
toposort with bfs,dfs and q
mst prims - implementation and q
mst - kruskal and dsu
dijkstra    https://leetcode.com/problems/path-with-minimum-effort/solutions/909002/java-python-3-3-codes-binary-search-bellman-ford-and-dijkstra-w-brief-explanation-and-analysis/?envType=daily-question&envId=2023-09-16
bellmann ford 
floyd warshall
kosaraju
tarjan
articulation point
blocks
bridges
clone graph 
2 color problem
smallest multiple with 0-1
functional graph
some q

//good code in class
A subgraph S of a graph G is a graph such that
• The vertices of S are a subset of the vertices of G
• The edges of S are a subset of the edges of G

• A spanning subgraph of G is a subgraph that contains all the vertices of G
• A forest is an undirected graph without cycles
• The connected components of a forest are trees

A connected component of a graph G is a maximal connected subgraph of G


The discovery edges labeled by BFS(G, s) form a spanning tree Ts of Gs in undirected graph, bfs not have back edges.

• Discovery and finish times have parenthesis structure
• represent discovery of u with left parenthesis "(u"
• represent finishin of u with right parenthesis "u)"
• history of discoveries and finishings makes a well-formed expression (parenthesis are properly nested)

• Intuition for proof: any two intervals are either disjoint or enclosed
• Overlaping intervals would mean finishing ancestor, before finishing descendant or starting descendant without starting ancestor

Detecting cycle in graph
--> undirected - normal dfs, bfs
--> directed - dfs with recursion stack, bfs with toposort

A directed graph G is acyclic iff a DFS of G yields no back edges

shortest path in unweighted graph - bfs
shortest path in DAG - toposort

// toposort with bfs
    vector<int> in_degree(V, 0); 
  
    for (int u = 0; u < V; u++) { 
        for (int x:adj[u]) 
            in_degree[x]++; 
    } 
  
    queue<int> q; 
    for (int i = 0; i < V; i++) 
        if (in_degree[i] == 0) 
            q.push(i); 

    int count=0;  
    while (!q.empty()) { 
        int u = q.front(); 
        q.pop(); 
  
        for (int x: adj[u]) 
            if (--in_degree[x] == 0) 
                q.push(x); 
        count++;
    } 
    if (count != V) { 
        cout << "There exists a cycle in the graph\n"; 
    }
    else{
        cout << "There exists no cycle in the graph\n";
    }

//toposort with dfs
    stack<int> st;
    
    for(int u=0;u<V;u++){
        if(visited[u]==false){
            DFS(adj,u,st,visited);
        }
    }
    void DFS(vector<int> adj[], int u,stack<int> &st, bool visited[]) 
    { 	
        visited[u]=true;
        
        for(int v:adj[u]){
            if(visited[v]==false)
                DFS(adj,v,st,visited);
        }
        st.push(u);
    }
    // reverse of stack gives ans




https://open.kattis.com/problems/quantumsuperposition
#include <bitset>
#include <iostream>
#include <queue>
#include <vector>
using namespace std;

int n[2], m[2];
vector<int> g[2][1001];
vector<int> back[2][1001];
bitset<2001> dp[2][1001];

void gen(int x) {
	int in_degree[1001] = {};
	for (int i = 0; i < m[x]; i++) {
		int a, b;
		cin >> a >> b;
		g[x][a].push_back(b);
		back[x][b].push_back(a);
		in_degree[b]++;
	}
	// finding length of routes of first universe
	queue<int> q;
	for (int i = 0; i <= n[x]; i++) {
		if (in_degree[i] == 0) { q.push(i); }
	}
	while (!q.empty()) {
		int node = q.front();
		q.pop();
		// using dp while processing the nodes topologically
		if (back[x][node].empty()) dp[x][node][0] = 1;
		for (int before : back[x][node]) dp[x][node] |= dp[x][before] << 1;
		for (int next : g[x][node]) {
			in_degree[next]--;
			if (in_degree[next] == 0) q.push(next);
		}
	}
}

int main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cin >> n[0] >> n[1] >> m[0] >> m[1];
	gen(0);
	gen(1);
	// preprocessing all possible sums between the universes
	bitset<2001> ans;
	for (int i = 0; i < 1001; i++)
		if (dp[0][n[0]][i]) ans |= dp[1][n[1]] << i;
	int Q;
	cin >> Q;
	for (int i = 0; i < Q; i++) {
		int a;
		cin >> a;
		if (ans[a]) {
			cout << "Yes" << endl;
		} else {
			cout << "No" << endl;
		}
	}
	return 0;
}

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long int
#define vi vector<ll>
#define fo(i,a,b) for(ll i = a; i <= b; i++)

int main() {

    fast;

    ll n1,n2,m1,m2;
    cin >> n1 >> n2 >> m1 >> m2;

    vi adj1[n1+5];
    vi indegree1(n1+5,0);
    set<ll> possible1[n1+5];

    vi adj2[n2+5];
    vi indegree2(n2+5,0);
    set<ll> possible2[n2+5];

    fo(i,1,m1){
        ll a,b;
        cin >> a >> b;

        indegree1[b]++;
        adj1[a].push_back(b);
    }

    fo(i,1,m2){
        ll a,b;
        cin >> a >> b;

        indegree2[b]++;
        adj2[a].push_back(b);
    }

    queue<ll> q1;

    q1.push(1);
    possible1[1].insert(0);

    while(!q1.empty()){
        ll a = q1.front();
        q1.pop();

        for(auto u : adj1[a]){
            for(auto r : possible1[a]){
                possible1[u].insert(r+1);
            }

            indegree1[u]--;
            if(indegree1[u] == 0) q1.push(u);
        }
    }

    queue<ll> q2;

    q2.push(1);
    possible2[1].insert(0);

    while(!q2.empty()){
        ll a = q2.front();
        q2.pop();

        for(auto u : adj2[a]){
            for(auto r : possible2[a]){
                possible2[u].insert(r+1);
            }

            indegree2[u]--;
            if(indegree2[u] == 0) q2.push(u);
        }
    }

    vector<bool> possible(2005,false);

    for(auto u : possible1[n1]){
        for(auto w : possible2[n2]){
            if(u+w <= 2001 ) possible[u+w] = true;
        }
    }

    ll q;
    cin >> q;

    while(q--){
        ll x;
        cin >> x;

        if(possible[x] == true) cout << "Yes\n";
        else cout << "No\n";
    }

    return 0;
}




  
//MST prims
it maintains two sets inMst and notinmst, it find the minimum weighted edge which connects both sets
implementation similar to dijkstra
invariat - prior to each iteration, A is subset of some minimum spanning tree

#include<bits/stdc++.h>
using namespace std;

void heapify(vpii &a,int i,int n,int *pos){
    int l=2*i+1;
    int r=2*i+2;
    int mini=i;
    if(a[l].ff<a[i].ff&&l<n){
        pos[a[i].ss]=l;
        pos[a[l].ss]=i;
        swap(a[i],a[l]);
        mini=l;
    }
    if(a[r].ff<a[i].ff&&r<n){
        pos[a[r].ss]=i;
        pos[a[i].ss]=r;
        swap(a[i],a[r]);
        mini=r;
    }
    
    if(mini!=i){
        heapify(a,mini,n,pos);
    }
}


signed main(){
    int n,m;
    cin>>n>>m;
    vvi adj(n);
    vvi w(n+1,vi (n+1,0));
    for(int i=0;i<m;i++){
        int u,v,weight;
        cin>>u>>v>>weight;
        adj[u].push_back(v);
        adj[v].push_back(u);
        w[u][v]=weight;
        w[v][u]=weight;
    }
    
    vi d(n,INT_MAX);
    int src=0;
    d[src]=0;
    
    vector<pair<int,int>>a(n);   //{weight, vertex no}
    int pos[n]={0};
    a[0].ff=0;
    a[0].ss=0;
    pos[a[0].ss]=0;
    for(int i=1;i<n;i++){
        a[i].ff=INT_MAX;
        a[i].ss=i;
        pos[a[i].ss]=i;
    }
    
    for(int i=n/2-1;i>=0;i--){
        heapify(a,i,n,pos);
    }
    int ans=0;
    
    int heap_size=n;
    for(int i=0;i<n;i++){
        //relax operation 
        // min heap
        pii u=a[0];
        int vertex_no=u.ss;
        int wt=u.ff;
        ans=ans+wt;
        for(auto x:adj[vertex_no]){
            int j=pos[x];
            int vertex_w=a[j].ff;
            int new_vertex_no=a[j].ss;
            if(w[vertex_no][new_vertex_no]<vertex_w){
                a[j].ff=w[vertex_no][new_vertex_no];   
                // while(j!=0){
                //     int p=(j-1)/2;
                //     if(a[p].ss<a[j].ss){
                //         pos[a[p].ff]=j;
                //         pos[a[j].ff]=p;
                //         swap(a[j],a[p]);
                //     }
                //     j=p;
                // }
            }
        }
        heap_size--;
        pos[a[heap_size].ss]=0;
        pos[a[0].ss]=heap_size;
        swap(a[0],a[heap_size]);
        heapify(a,0,heap_size,pos);
    }
    cout<<ans<<endl;
    for(int i=0;i<n;i++){
        cout<<a[i].ss<<" "<<a[i].ff<<endl;
    }

    return 0;
}

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

int prims(vector<vector<pii>>&adj,int &n){
    int ans=0;
    vector<int>weights(n,INT64_MAX);
    set<pii>s;
    weights[0]=0;
    s.insert({0,0});
    vector<int>visited(n,0);
    int cnt=0;
    while(s.size()>0){
        int u=s.begin()->second;
        s.erase(s.begin());
        cnt++;
        if(visited[u]==1) continue;
        visited[u]=1;
        ans+=weights[u];
        
        for(auto x:adj[u]){
            int v=x.first;
            int w=x.second;
            if(visited[v]==0&&w<weights[v]){
                
                
                weights[v]=w;
                s.insert({w,v});
            }
        }
    }
    if(cnt<n)return -1;
    return ans;
}

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,m;
    cin>>n>>m;
    vector<vector<pii>>adj(n);
    while(m--){
        int u,v,w;
        cin>>u>>v>>w;
        u--;v--;
        adj[u].push_back({v,w});
        adj[v].push_back({u,w});
    }
    int ans=prims(adj,n);
    cout<<ans<<endl;
    
    
    
    return 0;
}



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

int prims(vector<vector<pii>>&adj,int &n){
    int ans=0;
    vector<int>weights(n,INT64_MAX);
    set<pii>s;
    weights[0]=0;
    s.insert({0,0});
    vector<int>visited(n,0);
    int cnt=0;
    while(s.size()>0){
        int u=s.begin()->second;
        s.erase(s.begin());
        cnt++;
        if(visited[u]==1) continue;
        visited[u]=1;
        ans+=weights[u];
        
        for(auto x:adj[u]){
            int v=x.first;
            int w=x.second;
            if(visited[v]==0&&w<weights[v]){
                
                
                weights[v]=w;
                s.insert({w,v});
            }
        }
    }
    if(cnt<n)return -1;
    return ans;
}

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,m;
    cin>>n>>m;
    vector<vector<pii>>adj(n);
    while(m--){
        int u,v,w;
        cin>>u>>v>>w;
        u--;v--;
        adj[u].push_back({v,w});
        adj[v].push_back({u,w});
    }
    int ans=prims(adj,n);
    cout<<ans<<endl;
    
    
    
    return 0;
}



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



int prims(vector<vector<pii>>&adj,int &n){
    int ans=0;
    vector<int>weights(n,INT64_MAX);
    set<pii>s;
    weights[0]=0;
    s.insert({0,0});
    vector<int>visited(n,0);
    
    while(s.size()>0){
        auto it=s.end();
        // it--;
        it=s.begin();
        int u=it->second;
        s.erase(it);
        
        if(visited[u]==1) continue;
        visited[u]=1;
        
        
        for(auto x:adj[u]){
            int v=x.first;
            int w=x.second;
            if(visited[v]==0&&w<weights[v]){
                
                
                weights[v]=w;
                s.insert({w,v});
            }
        }
    }
    for(int i=0;i<n;i++) ans+=weights[i];
    
    return ans;
}

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n;
    cin>>n;
    int a[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    vector<vector<pii>>adj(n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(j==i) continue;
            adj[i].push_back({j,-(a[i]^a[j])});  //imp of brackets
        }
    }
    
    int ans=-prims(adj,n);
    cout<<ans<<endl;
    
    
    
    
    return 0;
}

//MST kruskal
sort all edges in incresing order
intialize mst as empty set 
while mst.size!=V-1
    if adding e to mst does not cause a cycle, then add it 

implemented using DSU

//union find/ disjoint set/ merge find
int find(int x) {
    while (x != link[x]) x = link[x];
    return x;
}

void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (size[a] < size[b]) swap(a,b);
    size[a] += size[b];
    link[b] = a;
}

// path compression 
If we call find_set(v) for some vertex v, we actually find the representative p 
for all vertices that we visit on the path between v and the actual representative p. 
The trick is to make the paths for all those nodes shorter, by setting the parent of each visited vertex 
directly to p.

//path compression with union by size / rank - 
we will reach nearly constant time queries. 
It turns out, that the final amortized time complexity is O(alpha(n)), where 
alpha(n) is the inverse Ackermann function, which grows very slowly. 
In fact it grows so slowly, that it doesnt exceed 4 for all reasonable n (approximately n < 10^600).

DSU with union by size / rank, but without path compression works in O(log n) time per query.
without union by rank is O(n)
push in bigger size (path compresion)

// moo tube
http://www.usaco.org/index.php?page=viewproblem2&cpid=789
#include <bits/stdc++.h>

using namespace std;

struct DSU {
	vector<int> e;
	void init(int n) { e = vector<int>(n, -1); }
	int get(int x) { return e[x] < 0 ? x : e[x] = get(e[x]); };
	bool sameSet(int x, int y) { return get(x) == get(y); };
	int size(int x) { return -e[get(x)]; }
	bool unite(int x, int y) {
		x = get(x), y = get(y);
		if (x == y) return false;
		if (e[x] > e[y]) swap(x, y);
		e[x] += e[y];
		e[y] = x;
		return true;
	}
};

bool cmp(const pair<int, pair<int, int>> &a,
         const pair<int, pair<int, int>> &b) {
	return a.second.second > b.second.second;
}

int main() {
	freopen("mootube.in", "r", stdin);
	freopen("mootube.out", "w", stdout);

	int n, q;
	cin >> n >> q;
	vector<pair<int, pair<int, int>>> edges(n - 1);
	for (int i = 0; i < n - 1; i++) {
		int u, v, w;
		cin >> u >> v >> w;
		u--;
		v--;
		edges[i] = make_pair(w, make_pair(u, v));
	}
	vector<pair<int, pair<int, int>>> queries(q);
	for (int i = 0; i < q; i++) {
		int v, k;
		cin >> k >> v;
		v--;
		queries[i] = make_pair(i, make_pair(v, k));
	}
	sort(queries.begin(), queries.end(), cmp);
	sort(edges.begin(), edges.end(), greater<pair<int, pair<int, int>>>());

	DSU dsu;
	dsu.init(n);
	vector<int> sol(q);
	int idx = 0;
	for (auto query : queries) {
		int v = query.second.first;
		int curK = query.second.second;
		while (idx < (int)edges.size() && edges[idx].first >= curK) {
			dsu.unite(edges[idx].second.first, edges[idx].second.second);
			idx++;
		}
		sol[query.first] = dsu.size(v) - 1;
	}

	for (auto x : sol) { cout << x << '\n'; }
}


https://cses.fi/problemset/task/1676

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
 
int find(int *parent,int *rank,int x){
    if(x==parent[x]){
        return x;
    }
    return parent[x]=find(parent,rank,parent[x]);
}
void unions(int *parent,int *rank,int x,int y){
    int p1=find(parent,rank,x);
    int p2=find(parent,rank,y);
    if(p1!=p2){
        if(rank[p1]<rank[p2]){
            parent[p1]=p2;
            rank[p2]+=rank[p1];
        }
        else{
            parent[p2]=p1;
            rank[p1]+=rank[p2];
        }
    }
}
 
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,m;
    cin>>n>>m;
    int parent[n];
    int rank[n];
    for(int i=0;i<n;i++){
        parent[i]=i;
        rank[i]=1;
    }
    
    
    int comp=n;
    int sz=1;
    while(m--){
        int x,y;
        cin>>x>>y;
        x--;
        y--;
        int p1=find(parent,rank,x);
        int p2=find(parent,rank,y);
        if(p1!=p2){
            comp--;
            unions(parent,rank,x,y);
            int p=find(parent,rank,x);
            sz=max(sz,rank[p]);
        }
        
        cout<<comp<<" "<<sz<<endl;
    }
    return 0;
}






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

int find(int *parent,int *rank,int x){
    if(x==parent[x]){
        return x;
    }
    return parent[x]=find(parent,rank,parent[x]);
}
void unions(int *parent,int *rank,int x,int y){
    int p1=find(parent,rank,x);
    int p2=find(parent,rank,y);
    if(p1!=p2){
        if(rank[p1]<rank[p2]){
            parent[p1]=p2;
            rank[p2]+=rank[p1];
        }
        else{
            parent[p2]=p1;
            rank[p1]+=rank[p2];
        }
    }
}

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,m;
    cin>>n>>m;
    int parent[n],rank[n];
    for(int i=0;i<n;i++){
        parent[i]=i;
        rank[i]=1;
    }
    
    int p[n];
    for(int i=0;i<n;i++){
        cin>>p[i];
        p[i]--;
    }
    vector<vector<int>>edges(m,vector<int>(3,0));
    for(int i=0;i<m;i++){
        cin>>edges[i][1]>>edges[i][2]>>edges[i][0];
        edges[i][1]--;
        edges[i][2]--;
    }
    sort(all(edges));
    int cnt=m;
    for(int i=0;i<n;i++){
        while(find(parent,rank,i)!=find(parent,rank,p[i])){
            cnt--;
            unions(parent,rank,edges[cnt][1],edges[cnt][2]);
        }
    }
    if(cnt==m){
        cout<<-1<<endl;
    }
    else{
        cout<<edges[cnt][0]<<endl;
    }

    return 0;
}

//dijkstra (for +ve weights only, greedy)
implement a minheap
all d[]={INT_MAX}
d[src]=0
put all d in heap 
while heap is not empty
extract min from heap     o(vlogv)
relax its adjacents       O(elogv)

for implementation
-> use trick like instead for finding key in pq, creates its copy and add to pq again O((E+V)logE)  actual-logV
-> use my method, in which I implemented heap and stores position in heap and in decrease key, 
    I travelled upward and swap positions

#include<bits/stdc++.h>
using namespace std;

void heapify(vpii &a,int i,int n,int *pos){
    int l=2*i+1;
    int r=2*i+2;
    int mini=i;
    if(a[l].ff<a[i].ff&&l<n){
        pos[a[i].ss]=l;
        pos[a[l].ss]=i;
        swap(a[i],a[l]);
        mini=l;
    }
    if(a[r].ff<a[i].ff&&r<n){
        pos[a[r].ss]=i;
        pos[a[i].ss]=r;
        swap(a[i],a[r]);
        mini=r;
    }
    
    if(mini!=i){
        heapify(a,mini,n,pos);
    }
}


signed main(){
    int n,m;
    cin>>n>>m;
    vvi adj(n);
    vvi w(n+1,vi (n+1,0));
    for(int i=0;i<m;i++){
        int u,v,weight;
        cin>>u>>v>>weight;
        adj[u].push_back(v);
        adj[v].push_back(u);
        w[u][v]=weight;
        w[v][u]=weight;
    }
    
    vi d(n,INT_MAX);
    int src=0;
    d[src]=0;
    
    vector<pair<int,int>>a(n);
    int pos[n]={0};
    a[0].ff=0;
    a[0].ss=0;
    pos[a[0].ss]=0;
    for(int i=1;i<n;i++){
        a[i].ff=INT_MAX;
        a[i].ss=i;
        pos[a[i].ss]=i;
    }
    
    for(int i=n/2-1;i>=0;i--){
        heapify(a,i,n,pos);
    }
    
    int heap_size=n;
    for(int i=0;i<n-1;i++){
        //relax operation 
        // min heap
        pii u=a[0];
        heap_size--;
        
        pos[a[heap_size].ss]=0;
        pos[a[0].ss]=heap_size;
        swap(a[0],a[heap_size]);
        heapify(a,0,heap_size,pos);
        
        int vertex_no=u.ss;
        int wt=u.ff;
        
        for(auto x:adj[vertex_no]){
            int j=pos[x];
            int vertex_w=a[j].ff;
            int new_vertex_no=a[j].ss;
            if(wt+w[vertex_no][new_vertex_no]<vertex_w){
                a[j].ff=wt+w[vertex_no][new_vertex_no];
                while(j!=0){
                    int p=(j-1)/2;
                    if(a[p].ss<a[j].ss){
                        pos[a[p].ff]=j;
                        pos[a[j].ff]=p;
                        swap(a[j],a[p]);
                    }
                    j=p;            
                }
            }
            
        }
    }
    for(int i=0;i<n;i++){
        cout<<a[i].ss<<" "<<a[i].ff<<endl;
    }    
    return 0;
}

vector<int> dijkstra(vector<vector<pair<int,int>>>&adj,int u,int &n){
    vector<int>dist(n,INT64_MAX);
    priority_queue<pii,vector<pii>,greater<pii>>pq;
    
    dist[u]=0;
    pq.push({0,u});
    
    while(pq.size()>0){
        int d=pq.top().first;
        int u=pq.top().second;
        pq.pop();
        
        if(d!=dist[u]) continue;
        
        for(auto x:adj[u]){
            int v=x.first;
            int w=x.second;
            if(w+dist[u]<dist[v]){
                dist[v]=w+dist[u];
                pq.push({dist[v],v});
            }
        }
        
    }
    return dist;
    
    
}

// lasers usaco
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
    int n;
    cin>>n;
    vector<pair<int,int>>points(n+2);
    map<int,vector<int>>mx;
    map<int,vector<int>>my;
    
    for(int i=0;i<n+2;i++){
        cin>>points[i].ff>>points[i].ss;
        mx[points[i].ff].push_back(i);
        my[points[i].ss].push_back(i);
    }
    queue<pair<int,int>>q;
    q.push({0,0});
    q.push({0,1});
    vector<int>dist(n+2,1e9);
    dist[0]=0;
    while(q.size()>0){
        int idx=q.front().ff;
        int is_y=q.front().ss;
        q.pop();
        if(is_y==0){
            int x=points[idx].ff;
            for(int new_idx:mx[x]){
                if(dist[new_idx]==1e9){
                    q.push({new_idx,1});
                    dist[new_idx]=1+dist[idx];
                }
            }
        }
        else{
            int y=points[idx].ss;
            for(int new_idx:my[y]){
                if(dist[new_idx]==1e9){
                    q.push({new_idx,0});
                    dist[new_idx]=1+dist[idx];
                }
            }
        }
    }
    cout << (dist[1] == 1e9 ? -1 : dist[1] - 1) << endl;
    
    
    return 0;
}

https://oj.uz/problem/view/BOI13_tracks

good way of applying bfs, diff connected componenets no need

Run flood fill to find each connected component with the same tracks.

Construct a graph where the nodes are the connected components and there are
edges between adjacent connected components.

The answer is the maximum distance from the node containing $(1, 1)$ to
another node. We can use BFS to find this distance.

#include <bits/stdc++.h>
using namespace std;

int dx[4]{1, -1, 0, 0}, dy[4]{0, 0, 1, -1};

int n, m, depth[4000][4000], ans = 1;
string snow[4000];

bool inside(int x, int y) {
	return (x > -1 && x < n && y > -1 && y < m && snow[x][y] != '.');
}

int main() {
	iostream::sync_with_stdio(false);
	cin.tie(0);

	cin >> n >> m;
	for (int i = 0; i < n; i++) cin >> snow[i];

	deque<pair<int, int>> q;
	q.push_back({0, 0});
	depth[0][0] = 1;

	while (q.size()) {
		pair<int, int> c = q.front();
		q.pop_front();
		ans = max(ans, depth[c.first][c.second]);

		for (int i = 0; i < 4; i++) {
			int x = c.first + dx[i], y = c.second + dy[i];
			if (inside(x, y) && depth[x][y] == 0) {
				if (snow[x][y] == snow[c.first][c.second]) {
					depth[x][y] = depth[c.first][c.second];
					q.push_front({x, y});
				} else {
					depth[x][y] = depth[c.first][c.second] + 1;
					q.push_back({x, y});
				}
			}
		}
	}

	cout << ans;
	return 0;
}



https://www.interviewbit.com/problems/min-cost-path/   use of dequeue instead of queue
how bfs used, no need of visited array
int Solution::solve(int m, int n, vector<string> &a) {
    vector<vector<int>> d(m,vector<int>(n,INT_MAX));
    deque<pair<int,int>> dq;
    dq.push_front({0,0});d[0][0] = 0;
    while(!dq.empty())
    {
        int i = dq.front().first,j = dq.front().second;
        cout<<i<<" "<<j<<endl;
        dq.pop_front();
        if(i)
        {
            int cost = 1;
            if(a[i][j] == 'U')cost = 0;
            if(d[i][j] + cost < d[i-1][j])
            {
                d[i-1][j] = d[i][j]+cost;
                if(cost)dq.push_back({i-1,j});
                else dq.push_front({i-1,j});
            }
        }
        if(j)
        {
            int cost = 1;
            if(a[i][j] == 'L')cost = 0;
            if(d[i][j] + cost < d[i][j-1])
            {
                d[i][j-1] = d[i][j]+cost;
                if(cost)dq.push_back({i,j-1});
                else dq.push_front({i,j-1});
            }
        }
        if(i+1 < m)
        {
            int cost = 1;
            if(a[i][j] == 'D')cost = 0;
            if(d[i][j] + cost < d[i+1][j])
            {
                d[i+1][j] = d[i][j]+cost;
                if(cost)dq.push_back({i+1,j});
                else dq.push_front({i+1,j});
            }
        }
        if(j+1 < n)
        {
            int cost = 1;
            if(a[i][j] == 'R')cost = 0;
            if(d[i][j] + cost < d[i][j+1])
            {
                d[i][j+1] = d[i][j]+cost;
                if(cost)dq.push_back({i,j+1});
                else dq.push_front({i,j+1});
            }
        }
    }
    return d[m-1][n-1];
}


no visited array, but with visited array its possible
int Solution::knight(int n, int m, int x1, int y1, int x2, int y2) {
    x1--;y1--;
    x2--;y2--;
    vector<vector<int>>moves(505,vector<int>(505,INT_MAX));
    moves[x1][y1]=0;
    queue<pair<int,int>>q;
    q.push({x1,y1});
    
    int dx[] = {-2, -2, -1, 1, 2, 2, 1, -1};
    int dy[] = {1, -1, -2, -2, -1, 1, 2, 2};

    while(q.size()>0){
        int x=q.front().first;
        int y=q.front().second;
        q.pop();
        for(int i=0;i<8;i++){
            if(x+dx[i]>=0&&x+dx[i]<n&&y+dy[i]>=0&&y+dy[i]<m){
                if(moves[x+dx[i]][y+dy[i]]>moves[x][y]+1){
                    moves[x+dx[i]][y+dy[i]]=min(moves[x+dx[i]][y+dy[i]],moves[x][y]+1);
                    q.push({x+dx[i],y+dy[i]});
                }
            }
        }
    }
    if(moves[x2][y2]==INT_MAX)
        return -1;
    else
    return moves[x2][y2];    
}

https://www.interviewbit.com/problems/useful-extra-edges/
Use dijkstra 2 times. Once with C as origin and another
with D as origin. Now for every node, you have the length of the shortest path from
C and from D. Now you can just iterate over each road and check if this road
helps you decreasing the existing distance between C and D.
whenever you get a better answer, you can update your answer to this
value and keep iterating on the auxillary roads.


https://www.interviewbit.com/problems/word-ladder-i/


int Solution::solve(string A, string B, vector<string> &C) {
    
    set<string>s;
    for(int i=0;i<C.size();i++){
        s.insert(C[i]);
    }
    
    map<string,int>m;
    m[A]=1;
    
    queue<string>q;
    q.push(A);
    
    
    while(q.size()>0){
        string u=q.front();
        q.pop();
        if(u==B){
            break;
        }
        string v=u;
        for(int i=0;i<u.size();i++){
            for(int j=0;j<26;j++){
                v=u;
                v[i]=j+'a';
                if(s.find(v)!=s.end()){
                    if(m.find(v)==m.end()){
                        m[v]=m[u]+1;
                        q.push(v);
                        s.erase(s.find(v));  // adding this pass testcase
                    }
                }
            }
        }
        
    }
    return m[B]; 
}
https://www.interviewbit.com/problems/word-ladder-ii/


vector<vector<string> > Solution::findLadders(string start, string end, vector<string> &dict) {
    set<string>s;
    for(int i=0;i<dict.size();i++){
        s.insert(dict[i]);
    }
    s.insert(start);
    s.insert(end);
    
    
    queue<vector<string>>q;
    vector<vector<string>>ans;
    
    if(start==end){
        ans.push_back({start});
        return ans;
    }
    
    int min_level=INT_MAX;
    int level=1;
    
    q.push({start});
    while(q.size()>0){
        vector<string>path=q.front();
        q.pop();
        
        string last=path.back();
        level=path.size();
        if(last==end){
            if(level<min_level){
                ans.clear();
                min_level=level;
                ans.push_back(path);
            }
            else if(level==min_level){
                ans.push_back(path);
            }
            else{
                break;
            }
        }
        
        set<string>visited=s;
        
        for(int i=0;i<path.size();i++){
            if(visited.find(path[i])!=visited.end()){
                visited.erase(visited.find(path[i]));
            }
        }
        
        string new_last=last;
        for(int i=0;i<last.size();i++){
            for(int j=0;j<26;j++){
                new_last=last;
                new_last[i]=j+'a';
                if(visited.find(new_last)!=visited.end()){
                    vector<string>temp=path;
                    temp.push_back(new_last);
                    q.push(temp);
                }
            }
        }
        
    }
    

    return ans;


}


// Bellmann ford (for -ve weights, for finding -ve cycles, dp)
relax all edges V-1 times
why V-1 times think of simple graph

//floyd warshall - all pair shortest path
#include<bits/stdc++.h>
using namespace std;
const int inf=1e6;         // infinity

void path(int i,int j,vector<int> &v,vector<vector<int>>&predecessor_matrix){
    v.push_back(j);
    
    if(predecessor_matrix[i][j]==0)                  // base case when from 'i' we reach back to 'i'
        return;
    
    path(i,predecessor_matrix[i][j],v,predecessor_matrix);     //helping function to find path
    
}

void find_path(int i,int j,vector<vector<int>>&predecessor_matrix){   // finding shortest path
    vector<int>v;
    path(i,j,v,predecessor_matrix);                 
    for(int i=v.size()-1;i>=0;i--){
        cout<<v[i]<<" ";                                         // printing path
    }
    cout<<endl;
}

int main(){        


    int v,e;
    cin>>v>>e;     // input: no of vertices and edges

    int weight_matrix[v+1][v+1];        

    for(int i=1;i<=v;i++){                // intializing weight matrix
        for(int j=1;j<=v;j++){
            if(i==j){
                weight_matrix[i][j]=0;               
            }
            else{
                weight_matrix[i][j]=inf;
            }

        }
    }
    
    vector<vector<int>> predecessor_matrix(v+1,vector<int>(v+1,0));

    for(int i=0;i<e;i++){
        int u,v,w;
        cin>>u>>v>>w;
        weight_matrix[u][v]=w;                     //taking input graph and intializing 
        predecessor_matrix[u][v]=u;                // weight and predecessor matrix
    }
    
    
    int shortest_dist[v+1][v+1];             // stores shortest diastance between two vertices
    
    for(int i=1;i<=v;i++){
        for(int j=1;j<=v;j++){
            shortest_dist[i][j]=weight_matrix[i][j];      // intially shortest distance is
        }                                                 // same as weight matrix
    }
     
    for(int k=1;k<=v;k++){                 //  k denotes the set {1,2,..,k} included 
        for(int i=1;i<=v;i++){              // from vertex i to j
            for(int j=1;j<=v;j++){
                if(shortest_dist[i][j]>shortest_dist[i][k]+shortest_dist[k][j]){  // we get shortest path
                                                                                   // using k th vertex
                    shortest_dist[i][j]=shortest_dist[i][k]+shortest_dist[k][j];
                    predecessor_matrix[i][j]=predecessor_matrix[k][j];
                
                }                                                                   // else shortest path
                    // using {1,2,..,k-1} vertex
            }
        }
    }

    for(int i=1;i<=v;i++){              // printing shortest distance between every pair 
        for(int j=1;j<=v;j++){
            if(shortest_dist[i][j]>=inf){
                cout<<"Not posiible from "<<i<<" to go "<<j<<endl;
                
            }
            else{
                cout<<"shortest distance from "<<i <<" to "<<j<<" is "<<shortest_dist[i][j]<<"   Following path ";
                find_path(i,j,predecessor_matrix);
                cout<<endl;
            }
        }
    }
}

// kosaraju (to find strongly connected components in directed graphs), two traversals 
// we can reach u from v and v from u
order the vertices in decreasing order of finish times in dfs  // one way - dfs topo stack, another way - finishtime array 
reverse all edges 
do dfs of above graph, all reachable vertices form strongly connected components

// tarjan algorithm (to find strongly connected components in directed graphs),uses low and disc time, one traversal
apply dfs, push elements in stack before exploring its childs
if adjacent of a vertex u is done with recursion, and disc[u]=low[u] then this vertex and all other vertex in the 
stack form strongly connected component
while updating low values, conside only back eges not cross edges
// find strongly connected components with 1 dfs
// kosaraju require 2 dfs.

Back edge: It is an edge (u, v) such that v is the ancestor of node u but is not part of the DFS tree.
Cross Edge: It is an edge that connects two nodes such that they do not have any ancestor and a descendant relationship between them.

#include<bits/stdc++.h>
using namespace std;

void dfs(vvi &adj,int *visited,int *disc,int *low,int s,int n,int &t,int p,int &root,vi &v){
    visited[s]=1;
    disc[s]=t;
    int r=0;
    t++;
    low[s]=disc[s];
    v.push_back(s);
    for(auto x:adj[s]){
        if(visited[x]==0){
            r++;
            dfs(adj,visited,disc,low,x,n,t,s,root,v);
        }
    }
    if(p==-1){
        if(r>1){
            root=2;
        }
    }
    if(r==0){
        for(auto x:adj[s]){
            if(x!=p)
            low[s]=min(low[s],disc[x]);
        }   
    }
    if(r>0){
        for(auto x:adj[s]){
            if(x!=p)
            low[s]=min(low[s],low[x]);
        }  
    }
    //cout<<s<<" "<<disc[s]<<" "<<low[s]<<endl;
    if(low[s]==disc[s]){
        while(v.back()!=s){
            cout<<v.back()<<" ";
            v.pop_back();
        }
        cout<<v.back();
        v.pop_back();
        cout<<endl;
        
    }    
}

signed main(){
    int n,m;
    cin>>n>>m;
    vvi adj(n);
    for(int i=0;i<m;i++){
        int u,v;
        cin>>u>>v;
        adj[u].push_back(v);
        //adj[v].push_back(u);
    }
    int disc[n]={0};
    int low[n]={0};
    for(int i=0;i<n;i++){
        low[i]=INT_MAX;    
    }   
    int visited[n]={0};
    int t=1;
    int root=0;
    vector<int>v;
    dfs(adj,visited,disc,low,0,n,t,-1,root,v);

    
    return 0;
}
int n; // number of nodes
vector<vector<int>> adj; // adjacency list of graph

vector<bool> visited;
vector<int> tin, low;
int timer;

void dfs(int v, int p = -1) {
    visited[v] = true;
    tin[v] = low[v] = timer++;
    for (int to : adj[v]) {
        if (to == p) continue;
        if (visited[to]) {
            low[v] = min(low[v], tin[to]);
        } else {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] > tin[v])
                IS_BRIDGE(v, to);
        }
    }
}

void find_bridges() {
    timer = 0;
    visited.assign(n, false);
    tin.assign(n, -1);
    low.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs(i);
    }
}
low[u]=min(  disc[u],   disc[v]  (u,v) is backedge,  low[v]  (u,v) is tree edge    )

// A C++ program to find strongly connected components in a
// given directed graph using Tarjan's algorithm (single
// DFS)
#include <bits/stdc++.h>
#define NIL -1
using namespace std;

// A class that represents an directed graph
class Graph {
	int V; // No. of vertices
	list<int>* adj; // A dynamic array of adjacency lists

	// A Recursive DFS based function used by SCC()
	void SCCUtil(int u, int disc[], int low[],
				stack<int>* st, bool stackMember[]);

public:
	Graph(int V); // Constructor
	void addEdge(int v,
				int w); // function to add an edge to graph
	void SCC(); // prints strongly connected components
};

Graph::Graph(int V)
{
	this->V = V;
	adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) { adj[v].push_back(w); }

// A recursive function that finds and prints strongly
// connected components using DFS traversal u --> The vertex
// to be visited next disc[] --> Stores discovery times of
// visited vertices low[] -- >> earliest visited vertex (the
// vertex with minimum
//			 discovery time) that can be reached from
//			 subtree rooted with current vertex
// *st -- >> To store all the connected ancestors (could be
// part
//		 of SCC)
// stackMember[] --> bit/index array for faster check
// whether
//				 a node is in stack
void Graph::SCCUtil(int u, int disc[], int low[],
					stack<int>* st, bool stackMember[])
{
	// A static variable is used for simplicity, we can
	// avoid use of static variable by passing a pointer.
	static int time = 0;

	// Initialize discovery time and low value
	disc[u] = low[u] = ++time;
	st->push(u);
	stackMember[u] = true;

	// Go through all vertices adjacent to this
	list<int>::iterator i;
	for (i = adj[u].begin(); i != adj[u].end(); ++i) {
		int v = *i; // v is current adjacent of 'u'

		// If v is not visited yet, then recur for it
		if (disc[v] == -1) {
			SCCUtil(v, disc, low, st, stackMember);

			// Check if the subtree rooted with 'v' has a
			// connection to one of the ancestors of 'u'
			// Case 1 (per above discussion on Disc and Low
			// value)
			low[u] = min(low[u], low[v]);
		}

		// Update low value of 'u' only of 'v' is still in
		// stack (i.e. it's a back edge, not cross edge).
		// Case 2 (per above discussion on Disc and Low
		// value)
		else if (stackMember[v] == true)
			low[u] = min(low[u], disc[v]);
	}

	// head node found, pop the stack and print an SCC
	int w = 0; // To store stack extracted vertices
	if (low[u] == disc[u]) {
		while (st->top() != u) {
			w = (int)st->top();
			cout << w << " ";
			stackMember[w] = false;
			st->pop();
		}
		w = (int)st->top();
		cout << w << "\n";
		stackMember[w] = false;
		st->pop();
	}
}

// The function to do DFS traversal. It uses SCCUtil()
void Graph::SCC()
{
	int* disc = new int[V];
	int* low = new int[V];
	bool* stackMember = new bool[V];
	stack<int>* st = new stack<int>();

	// Initialize disc and low, and stackMember arrays
	for (int i = 0; i < V; i++) {
		disc[i] = NIL;
		low[i] = NIL;
		stackMember[i] = false;
	}

	// Call the recursive helper function to find strongly
	// connected components in DFS tree with vertex 'i'
	for (int i = 0; i < V; i++)
		if (disc[i] == NIL)
			SCCUtil(i, disc, low, st, stackMember);
}

// Driver program to test above function
int main()
{
	cout << "\nSCCs in first graph \n";
	Graph g1(5);
	g1.addEdge(1, 0);
	g1.addEdge(0, 2);
	g1.addEdge(2, 1);
	g1.addEdge(0, 3);
	g1.addEdge(3, 4);
	g1.SCC();

	cout << "\nSCCs in second graph \n";
	Graph g2(4);
	g2.addEdge(0, 1);
	g2.addEdge(1, 2);
	g2.addEdge(2, 3);
	g2.SCC();

	cout << "\nSCCs in third graph \n";
	Graph g3(7);
	g3.addEdge(0, 1);
	g3.addEdge(1, 2);
	g3.addEdge(2, 0);
	g3.addEdge(1, 3);
	g3.addEdge(1, 4);
	g3.addEdge(1, 6);
	g3.addEdge(3, 5);
	g3.addEdge(4, 5);
	g3.SCC();

	cout << "\nSCCs in fourth graph \n";
	Graph g4(11);
	g4.addEdge(0, 1);
	g4.addEdge(0, 3);
	g4.addEdge(1, 2);
	g4.addEdge(1, 4);
	g4.addEdge(2, 0);
	g4.addEdge(2, 6);
	g4.addEdge(3, 2);
	g4.addEdge(4, 5);
	g4.addEdge(4, 6);
	g4.addEdge(5, 6);
	g4.addEdge(5, 7);
	g4.addEdge(5, 8);
	g4.addEdge(5, 9);
	g4.addEdge(6, 4);
	g4.addEdge(7, 9);
	g4.addEdge(8, 9);
	g4.addEdge(9, 8);
	g4.SCC();

	cout << "\nSCCs in fifth graph \n";
	Graph g5(5);
	g5.addEdge(0, 1);
	g5.addEdge(1, 2);
	g5.addEdge(2, 3);
	g5.addEdge(2, 4);
	g5.addEdge(3, 0);
	g5.addEdge(4, 2);
	g5.SCC();

	return 0;
}




//Articulation points for undirected, connnected graph
vertex whose removals increases the number of components
-> remove a vertex, then check no of connected components O((v+e)*(v+e))
-> if root of a dfs tree has 2 children , then it is articulation point O(v*(v+e))
-> if a non_root node u has a child v such that no ancestors are reachable from subtree rooted at v, then u->arti..
    if one such child which has no access to ancestor makes components increses
    leafs are never articulation point
    disc[u]= time at which node u is discovered
    low[u]= smallest discovery time reachable from u considering both types of edges
    A non root node u is articulation point if there exists a child v st 
        low[v]>=disc[u] 
            -<-
           |  |
     ---u-----v
     so u is arti..

    Acc to ranveer sir 
    a nonroot vertex v of a dfs tree is a cut vertex of G iff v is child s st there is no back edge from s to any 
    descendent of s to a proper ancestor of v 
    low(v)=min(disc(x)) x is vertex in dfs tree that can be reached from v by following zero or more tree
    edges followed by atmost one back edge 
#include<bits/stdc++.h>
using namespace std;

void dfs(vvi &adj,int *visited,int *disc,int *low,int s,int n,int &t,int p,int &root){
    visited[s]=1;
    disc[s]=t;
    int r=0;
    t++;
    
    for(auto x:adj[s]){
        if(visited[x]==0){
            r++;
            dfs(adj,visited,disc,low,x,n,t,s,root);
        }
    }
    if(p==-1){
        if(r>1){
            root=2;
        }
    }
    if(r==0){
        for(auto x:adj[s]){
            if(x!=p)
            low[s]=min(low[s],disc[x]);
        }   
    }
    if(r>0){
        for(auto x:adj[s]){
            if(x!=p)
            low[s]=min(low[s],low[x]);
        }  
    }
}

signed main(){
    int n,m;
    cin>>n>>m;
    vvi adj(n);
    for(int i=0;i<m;i++){
        int u,v;
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int disc[n]={0};
    int low[n]={0};
    for(int i=0;i<n;i++){
        low[i]=INT_MAX;    
    }   
    int visited[n]={0};
    int t=1;
    int root=0;
    dfs(adj,visited,disc,low,0,n,t,-1,root);
    for(int i=1;i<n;i++){
        if(low[i]>=disc[i]){
            cout<<i<<" ";
        }
        //cout<<i<<" "<<disc[i]<<" "<<low[i]<<endl;
    }
    if(root>1)
    cout<<"root also";
    
    
    
    
    return 0;
}
// C++ program to find articulation points in an undirected graph
#include <bits/stdc++.h>
using namespace std;

// A recursive function that find articulation
// points using DFS traversal
// adj[] --> Adjacency List representation of the graph
// u --> The vertex to be visited next
// visited[] --> keeps track of visited vertices
// disc[] --> Stores discovery times of visited vertices
// low[] -- >> earliest visited vertex (the vertex with minimum
// discovery time) that can be reached from subtree
// rooted with current vertex
// parent --> Stores the parent vertex in DFS tree
// isAP[] --> Stores articulation points
void APUtil(vector<int> adj[], int u, bool visited[],
			int disc[], int low[], int& time, int parent,
			bool isAP[])
{
	// Count of children in DFS Tree
	int children = 0;

	// Mark the current node as visited
	visited[u] = true;

	// Initialize discovery time and low value
	disc[u] = low[u] = ++time;

	// Go through all vertices adjacent to this
	for (auto v : adj[u]) {
		// If v is not visited yet, then make it a child of u
		// in DFS tree and recur for it
		if (!visited[v]) {
			children++;
			APUtil(adj, v, visited, disc, low, time, u, isAP);

			// Check if the subtree rooted with v has
			// a connection to one of the ancestors of u
			low[u] = min(low[u], low[v]);

			// If u is not root and low value of one of
			// its child is more than discovery value of u.
			if (parent != -1 && low[v] >= disc[u])
				isAP[u] = true;
		}

		// Update low value of u for parent function calls.
		else if (v != parent)
			low[u] = min(low[u], disc[v]);
	}

	// If u is root of DFS tree and has two or more children.
	if (parent == -1 && children > 1)
		isAP[u] = true;
}

void AP(vector<int> adj[], int V)
{
	int disc[V] = { 0 };
	int low[V];
	bool visited[V] = { false };
	bool isAP[V] = { false };
	int time = 0, par = -1;

	// Adding this loop so that the
	// code works even if we are given
	// disconnected graph
	for (int u = 0; u < V; u++)
		if (!visited[u])
			APUtil(adj, u, visited, disc, low,
				time, par, isAP);

	// Printing the APs
	for (int u = 0; u < V; u++)
		if (isAP[u] == true)
			cout << u << " ";
}

// Utility function to add an edge
void addEdge(vector<int> adj[], int u, int v)
{
	adj[u].push_back(v);
	adj[v].push_back(u);
}

int main()
{
	// Create graphs given in above diagrams
	cout << "Articulation points in first graph \n";
	int V = 5;
	vector<int> adj1[V];
	addEdge(adj1, 1, 0);
	addEdge(adj1, 0, 2);
	addEdge(adj1, 2, 1);
	addEdge(adj1, 0, 3);
	addEdge(adj1, 3, 4);
	AP(adj1, V);

	cout << "\nArticulation points in second graph \n";
	V = 4;
	vector<int> adj2[V];
	addEdge(adj2, 0, 1);
	addEdge(adj2, 1, 2);
	addEdge(adj2, 2, 3);
	AP(adj2, V);

	cout << "\nArticulation points in third graph \n";
	V = 7;
	vector<int> adj3[V];
	addEdge(adj3, 0, 1);
	addEdge(adj3, 1, 2);
	addEdge(adj3, 2, 0);
	addEdge(adj3, 1, 3);
	addEdge(adj3, 1, 4);
	addEdge(adj3, 1, 6);
	addEdge(adj3, 3, 5);
	addEdge(adj3, 4, 5);
	AP(adj3, V);

	return 0;
}


// Blocks/ biconnected components/ 2 connected components
a block is a maximally connected subgraph that has no cut vertex  
during dfs, use stack to store visited edges 
after we finish the recursive search of child u of v, if v is a cut vertex then output all edges from stack
when we reach root, output all whether it is cut vertex or not  

//Bridges in connected undirected graph
edges whose removal increases no of connected componenets
edge u-v is bridge if low[v]>disc[u] 
// it prints one edge two times 
#include<bits/stdc++.h>
using namespace std;

void dfs(vvi &adj,int *visited,int *disc,int *low,int s,int n,int &t,int p,int &root){
    visited[s]=1;
    disc[s]=t;
    int r=0;
    t++;
    
    for(auto x:adj[s]){
        if(visited[x]==0){
            r++;
            dfs(adj,visited,disc,low,x,n,t,s,root);
        }
    }
    if(p==-1){
        if(r>1){
            root=2;
        }
    }
    if(r==0){
        for(auto x:adj[s]){
            if(x!=p)
            low[s]=min(low[s],disc[x]);
        }   
    }
    if(r>0){
        for(auto x:adj[s]){
            if(x!=p)
            low[s]=min(low[s],low[x]);
        }  
    }   
}

signed main(){
    int n,m;
    cin>>n>>m;
    vvi adj(n);
    for(int i=0;i<m;i++){
        int u,v;
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int disc[n]={0};
    int low[n]={0};
    for(int i=0;i<n;i++){
        low[i]=INT_MAX;    
    }   
    int visited[n]={0};
    int t=1;
    int root=0;
    dfs(adj,visited,disc,low,0,n,t,-1,root);
    for(int i=0;i<n;i++){
        for(auto x:adj[i]){
            if(low[x]>disc[i]){
                cout<<i<<" "<<x<<endl;
            }
        }
    }

    return 0;
}
// A C++ program to find bridges in a given undirected graph
#include<bits/stdc++.h>
using namespace std;

// A class that represents an undirected graph
class Graph
{
	int V; // No. of vertices
	list<int> *adj; // A dynamic array of adjacency lists
	void bridgeUtil(int u, vector<bool>& visited, vector<int>& disc,
								vector<int>& low, int parent);
public:
	Graph(int V); // Constructor
	void addEdge(int v, int w); // to add an edge to graph
	void bridge(); // prints all bridges
};

Graph::Graph(int V)
{
	this->V = V;
	adj = new list<int>[V];
}

void Graph::addEdge(int v, int w)
{
	adj[v].push_back(w);
	adj[w].push_back(v); // Note: the graph is undirected
}

// A recursive function that finds and prints bridges using
// DFS traversal
// u --> The vertex to be visited next
// visited[] --> keeps track of visited vertices
// disc[] --> Stores discovery times of visited vertices
// parent[] --> Stores parent vertices in DFS tree
void Graph::bridgeUtil(int u, vector<bool>& visited, vector<int>& disc,
								vector<int>& low, int parent)
{
	// A static variable is used for simplicity, we can
	// avoid use of static variable by passing a pointer.
	static int time = 0;

	// Mark the current node as visited
	visited[u] = true;

	// Initialize discovery time and low value
	disc[u] = low[u] = ++time;

	// Go through all vertices adjacent to this
	list<int>::iterator i;
	for (i = adj[u].begin(); i != adj[u].end(); ++i)
	{
		int v = *i; // v is current adjacent of u
		
		// 1. If v is parent
		if(v==parent)
			continue;
	
		//2. If v is visited
		if(visited[v]){
		low[u] = min(low[u], disc[v]);
		}
	
		//3. If v is not visited
		else{
		parent = u;
		bridgeUtil(v, visited, disc, low, parent);

		// update the low of u as it's quite possible
		// a connection exists from v's descendants to u
		low[u] = min(low[u], low[v]);
		
		// if the lowest possible time to reach vertex v
		// is greater than discovery time of u it means
		// that v can be only be reached by vertex above v
		// so if that edge is removed v can't be reached so it's a bridge
		if (low[v] > disc[u])
			cout << u <<" " << v << endl;
		
		}
	}
}

// DFS based function to find all bridges. It uses recursive
// function bridgeUtil()
void Graph::bridge()
{
	// Mark all the vertices as not visited disc and low as -1
	vector<bool> visited (V,false);
	vector<int> disc (V,-1);
	vector<int> low (V,-1);
	
	// Initially there is no parent so let it be -1
	int parent = -1;

	// Call the recursive helper function to find Bridges
	// in DFS tree rooted with vertex 'i'
	for (int i = 0; i < V; i++)
		if (visited[i] == false)
			bridgeUtil(i, visited, disc, low, parent);
}

// Driver program to test above function
int main()
{
	// Create graphs given in above diagrams
	cout << "\nBridges in first graph \n";
	Graph g1(5);
	g1.addEdge(1, 0);
	g1.addEdge(0, 2);
	g1.addEdge(2, 1);
	g1.addEdge(0, 3);
	g1.addEdge(3, 4);
	g1.bridge();

	cout << "\nBridges in second graph \n";
	Graph g2(4);
	g2.addEdge(0, 1);
	g2.addEdge(1, 2);
	g2.addEdge(2, 3);
	g2.bridge();

	cout << "\nBridges in third graph \n";
	Graph g3(7);
	g3.addEdge(0, 1);
	g3.addEdge(1, 2);
	g3.addEdge(2, 0);
	g3.addEdge(1, 3);
	g3.addEdge(1, 4);
	g3.addEdge(1, 6);
	g3.addEdge(3, 5);
	g3.addEdge(4, 5);
	g3.bridge();

	return 0;
}



//clone graph
class Solution {
public:
    Node* cloneGraph(Node* node) {
        if(node==NULL){
            return NULL;
        }
        map<Node *,Node *>m;
        queue<Node *>q;
        q.push(node);
        map<Node *,int>visited;

        while(q.size()>0){
            Node* root=q.front();
            cout<<root->val<<endl;
            m[root]=new Node(root->val);
            q.pop();
            for(auto n:root->neighbors){
                if(visited[n]==0){
                    q.push(n);
                    visited[n]=1;
                }
            }
        }
        for(auto x:m){
            Node* intial=x.first;
            Node *final=x.second;
            for(auto x:intial->neighbors){
                (final->neighbors).push_back(m[x]);
            }
        }
        return m[node];
    }
};


// 2 color problem

void dfs(vector<vector<int>>&adj,vector<int>&visited,int s,int c,int *colour,int &r){
    visited[s]=1;
    colour[s]=c;
    for(auto v:adj[s]){
        if(visited[v]==-1){
            dfs(adj,visited,v,-c,colour,r);
        }
        else{
            if(colour[v]!=-c){
                r=1;
            }
        }
    }
    
}

int Solution::solve(int A, vector<vector<int> > &B) {
    vector<pair<int,int>>v(B.size());
    for(int i=0;i<B.size();i++){
        v[i].first=min(B[i][0],B[i][1]);
        v[i].second=max(B[i][0],B[i][1]);
    }
    sort(v.begin(),v.end());
    vector<vector<int>>adj(A+1);
    for(int i=0;i<B.size();i++){
        adj[v[i].first].push_back(v[i].second);
        adj[v[i].second].push_back(v[i].first);
    }
    vector<int>visited(A+1,-1);
    int colour[A+1]={0};
    int r=0;
    int c=0;
    for(int i=1;i<=A;i++){
        if(visited[i]==-1){
            c=1;
            dfs(adj,visited,i,c,colour,r);
        }
        
    }
    return 1-r;
    
    
}
https://www.interviewbit.com/problems/snake-ladder-problem/

//To handle the snakes and ladders, let’s keep an array go_immediately_to[] and initialize it like
// dont think ladder or snake as extra edges because they are must 
// use queue, I dont why it passed in interviewbit



https://www.interviewbit.com/problems/smallest-multiple-with-0-and-1/

string Solution::multiple(int n) {
    if(n==1){
        return "1";
    }
    queue<int>q;
    q.push(1);
    
    vector<int>v(n,-1);
    v[1]=1;
    
    vector<int>parent(n,-1);
    
    while(q.size()>0){
        int r=q.front();
        q.pop();
        
        if(r==0){
            break;
        }
        
        int r0=(r*10+0)%n;
        int r1=(r*10+1)%n;
          // if this reaminder exits int the vector means already stored value
        if(v[r0]==-1){
            v[r0]=0;
            q.push(r0);
            parent[r0]=r;
        }
        if(v[r1]==-1){
            v[r1]=1;
            q.push(r1);
            parent[r1]=r;
        }
    }
         // trace the path where we got 0 to the 1 initial from where we start then we will be reverse the string
     // when we loop untill rem==1
    string ans="";
    int rem=0;
    while(rem!=1){
        char c=v[rem]+'0';
        ans+=c;
        rem=parent[rem];
    }    
    ans+='1';
    reverse(ans.begin(),ans.end());
    return ans;
}

string Solution::multiple(int A) {
    int n = A;
    if(n==1) return "1";

    /*here we are setting the size of vector as 'n' because the number of different
    types f remainders cannot be more than 'n'
    example : for n=7 all the possible remainders can be : 0,1,2,3,4,5,6*/
    vector<long long> dp(n);
    dp[1]=1;
   
    //this queue stores the remainder of string with A
    queue<int> bfs;
    //for string ="1" remainder will be 1%A == 1, therefore
    bfs.push(1);
    while(!bfs.empty()){
        int r = bfs.front();
        bfs.pop();
        if(r==0) {
            string s;
            while(dp[r]){// explaination at bottom
                s += '0' + dp[r]%2;
                dp[r] /= 2;
            }
            reverse(s.begin(), s.end());
            return s;
        }
       
        for(int digit : {0,1}/*formed an array of 0&1*/){
            int new_r = (10*r + digit) % n;
            if(dp[new_r]==0){
                bfs.push(new_r);
               
                /* Here we are soring the string int binary format as the string
                is made of 0's and 1's . Example if the string form is "10" then in binary it is equal to 2
                "11" is 3, "110" is 6. Therefore we are storing the binary value of string in vector dp*/
                dp[new_r] = dp[r]*2 + digit;
            }
        }
    }
    return "-1";  
}


// Successor graph or functional graph
outdegree of each vertex is 1
    7
    ^ 
    |  
 1->3<-4

Floyd's Algorithm, also commonly referred to as the 
Tortoise and Hare Algorithm, is capable of detecting cycles in a functional graph in
O(N) time and O(1)memory (not counting the graph itself)

kth successor
As described briefly in CPH 16.3, the k-th successor of a certain node in a functional graph can be found in 
O(log k) time using binary jumping, given O(nlog u) time of preprocessing where 
u is the maximum length of each jump

Count Cycles
The following sample code counts the number of cycles in such a graph. 
The "stack" contains nodes that can reach the current node. If the current node points to a node v 
on the stack (on_stack[v] is true), then we know that a cycle has been created. 
However, if the current node points to a node v that has been previously visited but is not on the stack, 
then we know that the current chain of nodes points into a cycle that has already been considered.



https://codeforces.com/contest/1020/problem/B
bool cycle;
void solve(int i,vector<int>&ans,int *a){
    if(ans[i]>=-1){
        if(ans[i]==-1){
            ans[i]=i;
            cycle=true;
        }
        else{
            // i know where it is going to point
        }
        return ;
    }
    
    ans[i]=-1;
    solve(a[i],ans,a);
    if(ans[i]>=0){
        cycle=false;
        return;
    }
    if(cycle==true){
        ans[i]=i;
    }
    else{
        ans[i]=ans[a[i]];
    }
}

int main(){
    int n;
    cin>>n;
    int a[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
        a[i]--;
    }
    vector<int>ans(n,-2);
    
    for(int i=0;i<n;i++){
        solve(i,ans,a);
    }
    for(int i=0;i<n;i++){
        cout<<ans[i]+1<<" ";
    }
    cout<<endl;
    
    
    
    
    
    return 0;
}
//use floyd tortoise
https://leetcode.com/problems/count-visited-nodes-in-a-directed-graph/
class Solution {
public:
    int dfs(int i,vector<int>&edges,vector<int>&visited,int &start,int &end,vector<int>&path,int &t,vector<int>&ans){
        if(visited[i]>0){
            if(ans[i]>0){
                return ans[i]+1;
            }
            start=visited[i];
            end=t;
            return 0;
        }
        visited[i]=t;
        t++;
        path.push_back(i);
        ans[i]=dfs(edges[i],edges,visited,start,end,path,t,ans);
        if(ans[i]==0) return 0;
        return ans[i]+1;
    }
    
    vector<int> countVisitedNodes(vector<int>& edges) {
        
        int n=edges.size();
        vector<int>ans(n,0);
        vector<int>visited(n,0);
        int t=1;
        for(int i=0;i<n;i++){
            if(visited[i]==0){
                int start=-1,end=-1;
                vector<int>path;
                dfs(i,edges,visited,start,end,path,t,ans);
                if(ans[i]>0) continue;
                cout<<i<<endl;
                vector<int>non_cyc,cyc;
                for(int i=0;i<path.size();i++){
                    if(start<=visited[path[i]]&&visited[path[i]]<=end){
                        cyc.push_back(path[i]);
                    }
                    else{
                        non_cyc.push_back(path[i]);
                    }
                }
                
                int x=cyc.size();
                for(auto v:cyc){
                    ans[v]=x;
                }
                reverse(non_cyc.begin(),non_cyc.end());
                for(auto v:non_cyc){
                    x++;
                    ans[v]=x;
                }

            }
        }
        return ans;
    }
};

http://www.usaco.org/index.php?page=viewproblem2&cpid=740
wheres bessie
sometimes apply your logic, dont think of time complexity much

http://www.usaco.org/index.php?page=viewproblem2&cpid=992
always remember b search



moo network, some optimation req 
http://usaco.org/index.php?page=viewproblem2&cpid=1211

