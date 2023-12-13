// Contents:-
// BST and AVL trees 
// kth element in BST
// node ath kth distance from target node 
// rightmost node in last level of bt
// merge two bt 
// merge-k-sorted-lists
// burn a btree
// BST permutations 
// iterative traversals - pre,in,post
// line by line traversal method using null 
// left view (not typicall level order traversal)
// balance or not
// BST to DLL 
// zigzag path
// diameter - max distance b/w leaf nodes
// set next pointer using iterative traversals
// lca 
// max edge queries
// count node in o(logn*logn)
// theory - euler traversal, tree dp 
// tree dp - largest distance from every node, other q
// morris traversal - inorder in O(1) space
// subtree using euler path
// lca using euler path
// some q














//vertical traversal - use maps

full bt-0 or 2 childs
complete bt- all levels filled except last

A preorder tree walk processes each node before processing its children   
Ex printing hierarchy structure

A postorder tree walk processes each node after processing its children 
Ex memeory utilization

insert operation in preorder gives back tree

BST(left<node<right) insertion takes heigth O(n)

successor is leftmost node in the right subtree
successor is the lowest ancestor of x whose left child is also an ancestor of x.

deletion in tree
If x has two children, then to delete it, we have to find its successor (or predecessor) y
remove y
replace x with y



AVL Trees - A BST is perfectly balanced if, for every node, the difference between the
number of nodes in its left subtree and the number of nodes in its right subtree is
at most one

• Inserting a node, v, into an AVL tree changes the heights of some of the nodes in T.
• The only nodes whose heights can increase are the ancestors of node v. 
If insertion causes T to become unbalanced, then some ancestor of v would have a height imbalance.
• We travel up the tree from v until we find the first node x such that its grandparent z is 
unbalanced and y be the parent of node x


LL or RR single rotation
LR or RL double rotation
    z                          y
   / \        single rot      / \
   A   y         ->          z    x
      / \                   / \  / \
     B   x                 A  B  C  D
        / \
        C  D



    z                          x
   / \        double rot      / \
   A   y         ->          z    y
      / \                   / \  / \
     x   B                 A  C  D  B
    / \
    C  D



• The procedure of BST deletion of a node z:
• 1 child: delete it, connect child to parent
• 2 children: put successor in place of z, delete successor

Rebalancing after a Removal

• Let z be the first unbalanced node encountered while travelling up the tree from w (parent of removed
node) . Also, let y be the child of z with the larger height, and let x be the child of y with the larger height.


#include<bits/stdc++.h>
using namespace std;

struct node{
    int key;
    struct node *left;
    struct node *right;

    node(int k){
        key=k;
        left=NULL;
        right=NULL;
    }

    
};



//Given a binary search tree, write a function to find the kth smallest element in the tree.
int total(TreeNode* root){
    if(root==NULL){
        return 0;
    }   
    return 1+total(root->left)+total(root->right);
}
int Solution::kthsmallest(TreeNode* root, int k) {
    int left=total(root->left);
    int right=total(root->right);
    if(k==left+1){
        return root->val;
    }
    else if(k<=left){
        return Solution::kthsmallest(root->left,k);
    }
    else{
        return Solution::kthsmallest(root->right,k-left-1);
    }
}



// Given the root of a binary tree A, the value of a target node B, and an integer C, 
// return an array of the values of all nodes that have a distance C from the target node.

basic approach - create a graph, then do dfs/bfs from target vertex

https://www.interviewbit.com/problems/nodes-at-distance-k/
vector<int>ans;
void dfs(TreeNode *root,int k){
    if(root==NULL){
        return ;
    }
    if(k<0) return;
    if(k==0){
        ans.push_back(root->val);
    }
    else{
        dfs(root->left,k-1);
        dfs(root->right,k-1);
    }
}

void solve(TreeNode *root,int key,int k,int &d){
    if(root==NULL){
        return ;
    }
    if(root->val==key){
        dfs(root,k);
        d=0;
    }
    int l=-1;
    int r=-1;
    solve(root->left,key,k,l);
    solve(root->right,key,k,r);
    if(l>-1){
        d=l+1;
        if(d==k){
            ans.push_back(root->val);
        }
        else{
            dfs(root->right,k-d-1);
        }
    }
    else if(r>-1){
        d=r+1;
        if(d==k){
            ans.push_back(root->val);
        }
        else{
            dfs(root->left,k-d-1);
        }
    }
    
}
vector<int> Solution::distanceK(TreeNode* A, int B, int C) {
    ans.clear();
    int d=-1;
    solve(A,B,C,d);
    return ans;
}

int ans=0;

//You have to return the value of the rightmost node in the last level of the binary tree.

/*So, the idea here is to compare the height of the two subtrees.
If the height of the left subtree is higher, we go left, else we go right.
We can find the height of a subtree in O(log N), and this will be repeated for O(log N) steps.
Therefore, we can find the answer in O(log N * log N).*/


//merge 2 bt
TreeNode *MergeTrees(TreeNode * t1, TreeNode * t2) 
{ 
    if (!t1) 
        return t2; 
    if (!t2) 
        return t1; 
    t1->val += t2->val; 
    t1->left = MergeTrees(t1->left, t2->left); 
    t1->right = MergeTrees(t1->right, t2->right); 
    return t1; 
} 
TreeNode* Solution::solve(TreeNode* A, TreeNode* B) {
    assert( A && B);
    return MergeTrees(A,B);
}

https://leetcode.com/problems/merge-k-sorted-lists/

class Solution {
public:
    struct compare {
        bool operator()(const ListNode* l, const ListNode* r) {
            return l->val > r->val;
        }
    };
    ListNode *mergeKLists(vector<ListNode *> &lists) { //priority_queue
        priority_queue<ListNode *, vector<ListNode *>, compare> q;
        for(auto l : lists) {
            if(l)  q.push(l);
        }
        if(q.empty())  return NULL;

        ListNode* result = q.top();
        q.pop();
        if(result->next) q.push(result->next);
        ListNode* tail = result;            
        while(!q.empty()) {
            tail->next = q.top();
            q.pop();
            tail = tail->next;
            if(tail->next) q.push(tail->next);
        }
        return result;
    }
};




https://www.interviewbit.com/problems/burn-a-tree/
int sol(TreeNode *root,int key,int &d){
    if(root==NULL){
        return 0;
    }
    if(root->val==key){
        d=0;
        return 1;
    }
    int l=-1;
    int r=-1;
    int lh=sol(root->left,key,l);
    int rh=sol(root->right,key,r);
    if(l>-1){
        d=l+1;
        ans=max(ans,d+rh);
    }
    else if(r>-1){
        d=r+1;
        ans=max(ans,d+lh);
    }
    return max(lh,rh)+1;
}

int Solution::solve(TreeNode* A, int B) {
    ans=0;
    int d=-1;
    int h=sol(A,B,d);
    return ans;
}


// if that node is not a leaf node
class Solution {
public:
    int ans;
    int dfs(TreeNode* root,int &ans,int &d,int &key){
        if(root==NULL){
            return 0;
        }
        
        int l=-1,r=-1;
        int lh=dfs(root->left,ans,l,key);
        int rh=dfs(root->right,ans,r,key);
        if(root->val==key){
            d=0;
            ans=max(ans,max(lh,rh));
        }
        if(l>-1){
            d=l+1;
            ans=max(ans,rh+d);
        }
        if(r>-1){
            d=r+1;
            ans=max(ans,lh+d);
        }
        return max(lh,rh)+1;

    }
    int amountOfTime(TreeNode* root, int start) {
        if(root==NULL)
            return 0;

        ans=0;
        int d=-1;
        int h=dfs(root,ans,d,start);
        return ans;
        
    }
};


https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/
class Solution {
public:
    vector<vector<int>>c;
    int mod=1e9+7;
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
    int solve(vector<int>v){
        if(v.size()<=2){
            return 1;
        }
        int root=v[0];
        vector<int>left,right;
        for(int i=0;i<v.size();i++){
            if(v[i]<root){
                left.push_back(v[i]);
            }
            if(v[i]>root){
                right.push_back(v[i]);
            }
        }
        int l=left.size();
        int r=right.size();
        long long ans1=solve(left);
        long long ans2=solve(right);
        long long ans3=c[l+r][l];
        return (((ans1*ans2)%mod)*ans3)%mod;
    }

    int numOfWays(vector<int>& nums) {
        int n=nums.size();
        combination();
        return solve(nums)-1;
    }
};
another approach - make it a normal graph, do bfs from target vertex


https://www.interviewbit.com/problems/count-permutations-of-bst/  dp
def rec(N, M):
    if N<=1:
        if M==0: return 1
        return 0;

    ret=0

    for i=1 to N:
        x = i-1
        y = N-i

        ret1=0

        //iterate over height of left subtree
        for j = 0 to M-2:
            ret1 = ret1 + rec(x, j)*rec(y, M-1)

        //iterate over height of right subtree
        for j = 0 to M-2:
            ret1 = ret1 + rec(y, j)*rec(x, M-1)

        //add the case when both heights=M-1
        ret1 = ret1 + rec(x, M-1)*rec(y, M-1)

        ret = ret + ret1*choose(x+y, y)

    return ret


// inorder
vector<int> Solution::inorderTraversal(TreeNode* A) {
    vector<TreeNode *>v;
    vector<int>ans;
    
    TreeNode* root=A;
    
    while(root!=NULL||v.size()>0){
        while(root!=NULL){
            v.push_back(root);
            root=root->left;
        }
        root=v.back();
        v.pop_back();
        ans.push_back(root->val);
        root=root->right;
        
    }
    return ans;
    
}
// preorder
vector<int> Solution::preorderTraversal(TreeNode* A) {
    vector<TreeNode *>v;
    vector<int>ans;
    TreeNode *root=A;
    
    while(root!=NULL||v.size()>0){
        if(root==NULL){
            root=v.back();
            v.pop_back();
        }
        ans.push_back(root->val);
        if(root->right!=NULL){
            v.push_back(root->right);
        }
        root=root->left;
        
        
    }
    return ans;
    
    
}

// postorder
vector<int> Solution::postorderTraversal(TreeNode* A) {
    vector<TreeNode *>v;
    vector<int>ans;
    
    v.push_back(A);
    while(v.size()>0){
        TreeNode *root=v.back();
        v.pop_back();
        ans.push_back(root->val);
        if(root->left!=NULL){
            v.push_back(root->left);
        }
        if(root->right!=NULL){
            v.push_back(root->right);
        }
    }
    reverse(ans.begin(),ans.end());
    return ans;
}


// line by line
void levelorder(node *root){
    if(root!=NULL){
        queue<node *>q;
        q.push(root);
        q.push(NULL);
        while(q.size()>1){              // greater than 1 !!!!!!!!
            node *root=q.front();
            q.pop();

            if(root==NULL){
                cout<<endl;
                q.push(NULL);
                continue;
            }
            cout<<root->key<<" ";
            
            if(root->left!=NULL){
                q.push(root->left);
            }
            if(root->right!=NULL){
                q.push(root->right);
            }
        }



    }
}

// left view of tree recursive method printleft(A,0,1)
void printleft(node *root,int level,int &maxlevel){
    if(root==NULL)
        return ;

    if(level>maxlevel){
        maxlevel=level;
        cout<<root->key<<" ";
        
    }
    printleft(root->left,level+1,maxlevel);
    printleft(root->right,level+1,maxlevel);
}

//whether tree is balenced or not , how beautifully it handles see
int is_balence(node *root){
    if(root==NULL)
        return 0;

    int l=is_balence(root->left);
    int r=is_balence(root->right);

    if(l==-1||r==-1)
        return -1;

    if(abs(l-r)>1)
        return -1;
    else 
        return max(l,r)+1;
}

//great code to think upon it
node *b_to_dll(node *root){
    if(root==NULL)
        return root;
    
    static node *prev=NULL;
    node *head=b_to_dll(root->left);
    if(prev==NULL){
        head=root;
    }
    else{
        root->left=prev;
        prev->right=root;
    }
    prev=root;
    b_to_dll(root->right);
    return head;

}

void print_dll(node *root){
    while(root!=NULL){
        cout<<root->key<<" ";
        root=root->right;
    }
}




// print in zigzag order l to r then r to l

void printSpiral(Node *root){
    if (root == NULL) 
        return;  

    stack<Node*> s1;  
    stack<Node*> s2;  
  
    s1.push(root); 
  
    while (!s1.empty() || !s2.empty()) { 
        while (!s1.empty()) { 
            Node* temp = s1.top(); 
            s1.pop(); 
            cout << temp->key << " "; 
  
            if (temp->right) 
                s2.push(temp->right); 
            if (temp->left) 
                s2.push(temp->left); 
        } 
        
        while (!s2.empty()) { 
            Node* temp = s2.top(); 
            s2.pop(); 
            cout << temp->key << " "; 
  
            if (temp->left) 
                s1.push(temp->left); 
            if (temp->right) 
                s1.push(temp->right); 
        } 
    } 
}  

// diameter longest path b/w two leaf nodes
int height(node *root,int &d){
    if(root==NULL){
        return 0;
    }

    int l=height(root->left,d);
    int r=height(root->right,d);

    d=max(d,l+r+1);
    return max(l,r)+1;
}

int diameter_height(node *root){
    int d=0;
    int h=height(root,d);
    return d;
}

//Populating Next Right Pointers in Each Node

Node* connect(Node* root) {
    auto head = root;
    for(; root; root = root -> left) 
        for(auto cur = root; cur; cur = cur -> next)   // traverse each level - it's just BFS taking advantage of next pointers          
            if(cur -> left) {                          // update next pointers of children if they exist               
                cur -> left -> next = cur -> right;
                if(cur -> next) cur -> right -> next = cur -> next -> left;
            }
            else break;                                // if no children exist, stop iteration                                                  
    
    return head;
}

// LCA

// Approach 1, comparision of paths
int lca(node *root,int val1,int val2){

    vector<int>l;
    vector<int>r;

    bool x=find_path(root,val1,l);
    bool y=find_path(root,val2,r);

    for(int i=0;i<l.size()-1&&i<r.size()-1;i++){
        if(l[i+1]!=r[i+1]){
            return l[i];
        }
    }
    return -1;
}
// approach 2, both nodes exists
node * lca1(node *root,int val1,int val2){
    if(root==NULL){
        return root;
    }

    if(root->key==val1||root->key==val2){
        return root;
    }

    node *l=lca1(root->left,val1,val2);
    node *r=lca1(root->right,val1,val2);

    if(l!=NULL&&r!=NULL){
        return root;
    }
    else if(l!=NULL){
        return l;
    }
    else{
        return r;
    }
}

// when 1 node not exists??????
bool find(TreeNode* root,int key){
    if(root==NULL){
        return false;
    }
    if(root->val==key){
        return true;
    }
    return ((root->left!=NULL&&find(root->left,key))||(root->right!=NULL&&find(root->right,key)));
}

TreeNode* lca1(TreeNode* root,int a,int b){
    if(root==NULL){
        return root;
    }
    if(root->val==a||root->val==b){
        return root;
    }
    auto l=lca1(root->left,a,b);
    auto r=lca1(root->right,a,b);
    
    if(l!=NULL&&r!=NULL){
        return root;
    }
    else if(l!=NULL){
        return l;
    }
    else{
        return r;
    }
}

int Solution::lca(TreeNode* A, int B, int C) {
    
    if(find(A,B)==true&&find(A,C)==true){
        TreeNode* l=lca1(A,B,C);
        if(l==NULL){
            return -1;
        }
        else{
            return l->val;
        }
    }
    else{
        return -1;
    }
    
}
// approach 4
find diff of heights, move that diff up a node 
query-O(logn)

However, we use a different tree traversal array than before: we add each
node to the array always when the depth-first search walks through the node,
and not only at the first visit. Hence, a node that has k children appears k+1
times in the array and there are a total of 2n-1 nodes in the array.

n find the lowest common ancestor of nodes a and b by finding the
node with the minimum depth between nodes a and b in the array
query-O(1)


distance (a,b)= depth(a)+depth(b)-2*depth(lca(a,b))



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

void dfs(vector<vector<int>>&adj,vector<vector<int>>&up,vector<int>&start,vector<int>&end,int &t,int u,int p,int m){
    start[u]=t;
    up[u][0]=p;
    for(int i=1;i<=m;i++){
        up[u][i]=up[up[u][i-1]][i-1];
    }
    for(auto v:adj[u]){
        if(v!=p){
            t++;
            dfs(adj,up,start,end,t,v,u,m);
        }
    }
    end[u]=t;
}
bool is_ancestor(int u,int v, vector<int>&start,vector<int>&end){
    return start[u]<=start[v]&&end[u]>=end[v];
}

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,q;
    cin>>n>>q;
    vector<vector<int>>adj(n);
    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        u--;v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int>height(n,0);
    vector<int>start(n,0);
    vector<int>end(n,0);
    int m=ceil(log2(n));
    vector<vector<int>>up(n,vector<int>(m+1,0));
    int t=1;
    dfs(adj,up,start,end,t,0,0,m);
    
    while(q--){
        int u,v;
        cin>>u>>v;
        u--;v--;
        int lca=-1;
        if(is_ancestor(u,v,start,end)){
            lca=u;
        }
        else if(is_ancestor(v,u,start,end)){
            lca=v;
        }
        else{
            for(int i=m;i>=0;i--){
                if(!is_ancestor(up[u][i],v,start,end)){
                    u=up[u][i];
                }
            }
            lca=up[u][0];
        }
        cout<<lca<<endl;
    }
    
    
    return 0;
}
//cses company queries 2
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
int jump(int u,int k,vector<vector<int>>&up){
    for(int i=0;i<31;i++){
        if(k&(1<<i)){
            u=up[u][i];
            if(u==-1){
                break;
            }
        }
    }
    return u;
}
int lca(int u,int v,vector<vector<int>>&up,vector<int>&depth){
    if(depth[u]<depth[v]){
        swap(u,v);
    }
    u=jump(u,depth[u]-depth[v],up);
    if(u==v){
        return u;
    }
    for(int i=30;i>=0;i--){
        int ances_u=up[u][i];
        int ances_v=up[v][i];
        
        if(ances_u!=ances_v){
            u=ances_u;
            v=ances_v;
        }
    }
    return up[u][0];
}
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,q;
    cin>>n>>q;
    vector<vector<int>>up(n,vector<int>(31,-1));
    vector<int>depth(n,0);
    for(int i=1;i<n;i++){
        cin>>up[i][0];
        up[i][0]--;
        depth[i]=1+depth[up[i][0]];
    }
    for(int j=1;j<31;j++){
        for(int i=0;i<n;i++){
            if(up[i][j-1]==-1){
                up[i][j]=-1;
            }
            else{
                up[i][j]=up[up[i][j-1]][j-1];
            }
        }
    }
    while(q--){
        int u,v;
        cin>>u>>v;
        u--;v--;
        cout<<lca(u,v,up,depth)+1<<endl;   
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
int jump(int u,int k,vector<vector<int>>&up){
    for(int i=0;i<31;i++){
        if(k&(1<<i)){
            u=up[u][i];
            if(u==-1){
                break;
            }
        }
    }
    return u;
}
bool ances(int u,int v,vector<int>&start,vector<int>&end){
    if(u==-1)
        return true;
    return ((start[u]<=start[v])&&(end[u]>=end[v]));
}

int lca(int u,int v,vector<vector<int>>&up,vector<int>&start,vector<int>&end){
    if(ances(u,v,start,end))
        return u;
    if(ances(v,u,start,end))
        return v;
        
    for(int i=30;i>=0;i--){
        if(ances(up[u][i],v,start,end)==false){
            u=up[u][i];
        }
    }
    return up[u][0];
}

void euler_path(int u,int p,int &t,vector<int>&start,vector<int>&end,vector<vector<int>>&adj){
    start[u]=t;
    for(auto v:adj[u]){
        if(v!=p){
            t++;
            euler_path(v,u,t,start,end,adj);
        }
    }
    end[u]=t;
}
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,q;
    cin>>n>>q;
    vector<vector<int>>up(n,vector<int>(31,-1));
    vector<vector<int>>adj(n);
    
    for(int i=1;i<n;i++){
        cin>>up[i][0];
        up[i][0]--;
        adj[i].push_back(up[i][0]);
        adj[up[i][0]].push_back(i);
    }
    
    vector<int>start(n),end(n);
    int t=1;
    euler_path(0,-1,t,start,end,adj);
    for(int j=1;j<31;j++){
        for(int i=0;i<n;i++){
            if(up[i][j-1]==-1){
                up[i][j]=-1;
            }
            else{
                up[i][j]=up[up[i][j-1]][j-1];
            }
        }
    }
    while(q--){
        int u,v;
        cin>>u>>v;
        u--;v--;
        cout<<lca(u,v,up,start,end)+1<<endl;
        
        
    }
    return 0;
}



https://www.interviewbit.com/problems/max-edge-queries/
void dfs(vector<vector<pair<int,int>>>&adj,vector<vector<pair<int,int>>>&up,vector<int>&start,vector<int>&end,int &t,int u,int p,int m,int w){
    start[u]=t;
    up[u][0].first=p;
    up[u][0].second=w;
    for(int i=1;i<=m;i++){
        up[u][i].first=up[up[u][i-1].first][i-1].first;
        up[u][i].second=max(up[u][i-1].second,up[up[u][i-1].first][i-1].second);
    }
    for(auto x:adj[u]){
        int v=x.first;
        int w=x.second;
        if(v!=p){
            t++;
            dfs(adj,up,start,end,t,v,u,m,w);
        }
    }
    end[u]=t;
}
bool is_ancestor(int u,int v, vector<int>&start,vector<int>&end){
    return start[u]<=start[v]&&end[u]>=end[v];
}
int query(int u,int v,vector<int>&start,vector<int>&end,vector<vector<pair<int,int>>>&up,int &m){
    int ans=0;
    if(is_ancestor(v,u,start,end)){                         // no_negation
        for(int i=m;i>=0;i--){
            if(is_ancestor(v,up[u][i].first,start,end)){
                ans=max(ans,up[u][i].second);
                u=up[u][i].first;
            }
        }
        return ans;
    }
    if(is_ancestor(u,v,start,end)){
        return ans;
    }
    for(int i=m;i>=0;i--){  
        if(!is_ancestor(up[u][i].first,v,start,end)){       //negation
            ans=max(ans,up[u][i].second);
            u=up[u][i].first;
        }
    }
    ans=max(ans,up[u][0].second);
    return ans;
    
}

vector<int> Solution::solve(vector<vector<int> > &A, vector<vector<int> > &B) {
    int n,q;
    n=A.size()+1;
    q=B.size();
    vector<vector<pair<int,int>>>adj(n);
    for(int i=0;i<n-1;i++){
        int u,v,w;
        u=A[i][0];
        v=A[i][1];
        w=A[i][2];
        u--;v--;
        adj[u].push_back({v,w});
        adj[v].push_back({u,w});
    }
    vector<int>height(n,0);
    vector<int>start(n,0);
    vector<int>end(n,0);
    int m=ceil(log2(n));
    vector<vector<pair<int,int>>>up(n,vector<pair<int,int>>(m+1,{0,0}));
    int t=1;
    dfs(adj,up,start,end,t,0,0,m,0);
    
    vector<int>ans(q,0);
    
    for(int j=0;j<q;j++){
        int u,v;
        u=B[j][0];
        v=B[j][1];
        u--;v--;
        
        ans[j]=max(query(u,v,start,end,up,m),query(v,u,start,end,up,m));
        
    }
    
    return ans;


}

// counting number of nodes in complete BT , O(logn*logn)
int countNode(Node *root){
    int lh=0,rh=0;
    Node *curr=root;
    while(curr!=NULL){
        lh++;
        curr=curr->left;
    }
    curr=root;
    while(curr!=NULL){
        rh++;
        curr=curr->right;
    }
    if(lh==rh){
        return pow(2,lh)-1;
    }else{
        return 1+countNode(root->left)+countNode(root->right);
    }
}  



Some properties/definitions of trees:

A graph is a tree iff it is connected and contains N nodes and N-1 edges
A graph is a tree iff every pair of nodes has exactly one simple path between them
A graph is a tree iff it is connected and does not contain any cycles

A star graph has two common definitions. Try to understand what they mean - they typically appear in subtasks.
Definition 1: Only one node has degree greater than 1
Definition 2: Only one node has degree greater than 2

The ancestors of a node are its parent and parents ancestors

Typically, a node is considered its own ancestor as well (such as in the subtree definition)

The subtree of a node n are the set of nodes that have n as an ancestor

A node is typically considered to be in its own subtree

// Euler traversal
If we can preprocess a rooted tree such that every subtree corresponds to a contiguous range on an array,
we can do updates and range queries on it!

        1
      /   \
    2      3
         /   \
        4    5

node index  | 1 | 2 | 3 | 4 | 5 |
start time  | 0 | 1 | 2 | 3 | 4 |
end time    | 5 | 2 | 5 | 4 | 5 |


each range {start[i],start[i]} contains all ranges {start[j],start[j]} for each j in subtree of i
end[i]-start[i] =subtree size(i)

It is a common technique to calculate two DP arrays for some DP on trees problems.
Usually one DP array is responsible for calculating results within the subtree rooted at i. 
The other DP array calculates results outside of the subtree rooted at i.


// largest distance from every node
#include<bits/stdc++.h>
using namespace std;

void eval_depth(vector<vector<int>>&adj,vector<int>&depth,int u,int p){
    int h=0;
    for(auto v:adj[u]){
        if(v!=p){
            eval_depth(adj,depth,v,u);
            h=max(h,1+depth[v]);
        }
    }
    depth[u]=h;
}
void dfs(vector<vector<int>>&adj,vector<int>&depth,vector<int>&ans,int u,int p,int par_ans){
    vector<int>pre,suf;
    for(auto v:adj[u]){
        if(v!=p){
            pre.push_back(depth[v]);
            suf.push_back(depth[v]);
        }
    }
    for(int i=1;i<pre.size();i++){
        pre[i]=max(pre[i],pre[i-1]);
    }
    for(int i=suf.size()-2;i>=0;i--){
        suf[i]=max(suf[i],suf[i+1]);
    }
    int c=0;
    for(auto v:adj[u]){
        if(v!=p){
            int op1=INT_MIN,op2=INT_MIN;
            if(c>0) op1=pre[c-1];
            if(c<suf.size()-1) op2=suf[c+1];
            int partial_ans=1+max({par_ans,op1,op2});
            dfs(adj,depth,ans,v,u,partial_ans);
            c++;
        }
    }
    int x=-1;
    if(pre.size()>0) x=pre.back();
    ans[u]=1+max(par_ans,x);
    
}

signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n;
    cin>>n;
    vector<vector<int>>adj(n);
    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        u--;v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int>depth(n,0);
    vector<int>ans(n,0);
    eval_depth(adj,depth,0,-1);
    dfs(adj,depth,ans,0,-1,-1);
    for(int i=0;i<n;i++){
        cout<<ans[i]<<" ";
    }
    cout<<endl;
    
    return 0;
}

//subtree queries
https://cses.fi/problemset/task/1137
subtree queries
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
void euler_tour(vector<vector<int>>&adj,vector<int>&start,vector<int>&end,int u,int p,int &t){
    start[u]=t;
    for(auto v:adj[u]){
        if(v!=p){
            t++;
            euler_tour(adj,start,end,v,u,t);
        }
    }
    end[u]=t;
}


signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,q;
    cin>>n>>q;
    vector<int>bit(n+1,0);
	vector<int>a(n+1,0);
    for(int i=1;i<=n;i++){
        cin>>a[i];
    }
    vector<vector<int>>adj(n+1);
    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    vector<int>start(n+1,0);
    vector<int>end(n+1,0);
    int t=1;
    euler_tour(adj,start,end,1,-1,t);
    // for(int i=1;i<=n;i++){
    //     cout<<start[i]<<" "<<end[i]<<endl;
    // }
    for(int i=1;i<=n;i++){
        update(start[i],a[i],bit,n);
    }
    
    
    while(q--){
        int x;
        cin>>x;
        if(x==1){
            int idx,new_val;
            cin>>idx>>new_val;
            int old_val=a[idx];
            int diff=new_val-old_val;
            a[idx]=new_val;
            update(start[idx],diff,bit,n);
            
        }
        else{
            int idx;
            cin>>idx;
            cout<<sum_q(start[idx],end[idx],bit)<<endl;
        }
    }

    return 0;
}
// path queries
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
void euler_tour(vector<vector<int>>&adj,vector<int>&start,vector<int>&end,int u,int p,int &t){
    start[u]=t;
    for(auto v:adj[u]){
        if(v!=p){
            t++;
            euler_tour(adj,start,end,v,u,t);
        }
    }
    end[u]=t;
}
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,q;
    cin>>n>>q;
    vector<int>bit(n+1,0);
	vector<int>a(n+1,0);
    for(int i=1;i<=n;i++){
        cin>>a[i];
      
    }
    vector<vector<int>>adj(n+1);
    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    vector<int>start(n+1,0);
    vector<int>end(n+1,0);
    int t=1;
    euler_tour(adj,start,end,1,-1,t);
    // for(int i=1;i<=n;i++){
    //     cout<<start[i]<<" "<<end[i]<<endl;
    // }
    for(int i=1;i<=n;i++){
        update(start[i],a[i],bit,n);
        update(end[i]+1,-a[i],bit,n);
    }
    while(q--){
        int x;
        cin>>x;
        if(x==1){
            int idx,new_val;
            cin>>idx>>new_val;
            int old_val=a[idx];
            int diff=new_val-old_val;
            a[idx]=new_val;
            update(start[idx],diff,bit,n);
            update(end[idx]+1,-diff,bit,n);
            
        }
        else{
            int idx;
            cin>>idx;
            cout<<query(start[idx],bit)<<endl;
        }
    }
    return 0;
}
https://cses.fi/problemset/task/2134  
path queries 2 use advanced data structures


https://cses.fi/problemset/task/1130
tree matching dp on trees

dp_0[v] represent the maximum matching of the subtree of v such that we dont take any edges leading to 
some child of v.

dp_1[v] represent the maximum matching of the subtree of v such that we take one edge
leading into a child of v.

#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using vi = vector<int>;
#define pb push_back
#define rsz resize
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
using pi = pair<int, int>;
#define f first
#define s second
#define mp make_pair
void setIO(string name = "") {  // name is nonempty for USACO file I/O
	ios_base::sync_with_stdio(0);
	cin.tie(0);  // see Fast Input & Output
	if (sz(name)) {
		freopen((name + ".in").c_str(), "r", stdin);  // see Input & Output
		freopen((name + ".out").c_str(), "w", stdout);
	}
}

vi adj[200005];
int dp[200005][2];

void dfs(int v, int p) {
	for (int to : adj[v]) {
		if (to != p) {
			dfs(to, v);
			dp[v][0] += max(dp[to][0], dp[to][1]);
		}
	}
	for (int to : adj[v]) {
		if (to != p) {
			dp[v][1] = max(dp[v][1], dp[to][0] + 1 + dp[v][0] - max(dp[to][0], dp[to][1]));
		}
	}
}

int main() {
	setIO();
	int n;
	cin >> n;
	for (int i = 0; i < n - 1; i++) {
		int u, v;
		cin >> u >> v;
		u--, v--;
		adj[u].pb(v), adj[v].pb(u);
	}
	dfs(0, -1);
	cout << max(dp[0][0], dp[0][1]) << '\n';

}
greedy approach is also there 
int n;
vi adj[MX];
bool done[MX];
int ans = 0;

void dfs(int pre, int cur) {
	for (int i : adj[cur]) {
		if (i != pre) {
			dfs(cur, i);
			if (!done[i] && !done[cur]) done[cur] = done[i] = 1, ans++;
		}
	}
}

int main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cin >> n;
	F0R(i, n - 1) {
		int a, b;
		cin >> a >> b;
		adj[a].pb(b), adj[b].pb(a);
	}
	dfs(0, 1);
	cout << ans;
}

https://atcoder.jp/contests/dp/tasks/dp_p
dp at graphs
#include <iostream>
#include <vector>
using namespace std;

int MOD = 1000000007;
long dp[2][100000];
vector<int> adj[100000];
void dfs(int i, int p) {
	dp[0][i] = 1;
	dp[1][i] = 1;
	for (int v : adj[i]) {
		if (v != p) {
			dfs(v, i);
			dp[0][i] = (dp[0][i] * (dp[0][v] + dp[1][v])) % MOD;
		}
	}
	for (int v : adj[i]) {
		if (v != p) { dp[1][i] = (dp[1][i] * dp[0][v]) % MOD; }
	}
}
int main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	int n;
	cin >> n;
	for (int i = 0; i < n - 1; i++) {
		int u, v;
		cin >> u >> v;
		u--, v--;
		adj[u].push_back(v), adj[v].push_back(u);
	}
	dfs(0, -1);
	cout << (dp[0][0] + dp[1][0]) % MOD << '\n';
	return 0;
}

// morris traversal inorder in O(1) space
If current does not have left child

    a. Add current’s value

    b. Go to the right, i.e., current = current.right

Else

    a. In currents left subtree, make current the right child of the rightmost node

    b. Go to this left child, i.e., current = current.left

    
          1
        /   \
       2     3
      / \   /
     4   5 6

         2
        / \
       4   5
            \
             1
              \
               3
              /
             6

        4
         \
          2
           \
            5
             \
              1
               \
                3
               /
              6



        

https://leetcode.com/problems/binary-tree-inorder-traversal/

https://cses.fi/problemset/result/6344483/
// subtree queries
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
void euler_tour(vector<vector<int>>&adj,vector<int>&start,vector<int>&end,int u,int p,int &t){
    start[u]=t;
    for(auto v:adj[u]){
        if(v!=p){
            t++;
            euler_tour(adj,start,end,v,u,t);
        }
    }
    end[u]=t;
}
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,q;
    cin>>n>>q;
    vector<int>bit(n+1,0);
	vector<int>a(n+1,0);
    for(int i=1;i<=n;i++){
        cin>>a[i];
      
    }
    vector<vector<int>>adj(n+1);
    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    vector<int>start(n+1,0);
    vector<int>end(n+1,0);
    int t=1;
    euler_tour(adj,start,end,1,-1,t);
    // for(int i=1;i<=n;i++){
    //     cout<<start[i]<<" "<<end[i]<<endl;
    // }
    for(int i=1;i<=n;i++){
        update(start[i],a[i],bit,n);
    }
    while(q--){
        int x;
        cin>>x;
        if(x==1){
            int idx,new_val;
            cin>>idx>>new_val;
            int old_val=a[idx];
            int diff=new_val-old_val;
            a[idx]=new_val;
            update(start[idx],diff,bit,n);
            
        }
        else{
            int idx;
            cin>>idx;
            cout<<sum_q(start[idx],end[idx],bit)<<endl;
        }
    }
    return 0;
}

// lca using euler path

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
void dfs(vector<vector<int>>&adj,int u,int h,int p,vector<int>&euler,vector<int>&height,vector<int>&first){
    height[u]=h;
    first[u]=euler.size();
    euler.push_back(u);
    for(auto v:adj[u]){
        if(v!=p){
            dfs(adj,v,h+1,u,euler,height,first);
            euler.push_back(u);
        }
    }
}
void build(vector<int>&euler,vector<int>&tree,int node, int st,int end,vector<int>&first){
    if(st==end){
        tree[node]=euler[st];
        return;
    }
    int mid=(st+end)/2;
    build(euler,tree,2*node,st,mid,first);
    build(euler,tree,2*node+1,mid+1,end,first);
    if(tree[2*node]!=-1&&tree[2*node+1]!=-1){
        if(first[tree[2*node]]<first[tree[2*node+1]]){
            tree[node]=tree[2*node];
        }
        else{
            tree[node]=tree[2*node+1];
        }
    }
    else{
        tree[node]=max(tree[2*node],tree[2*node+1]);
    }
}
int query(vector<int>&euler,vector<int>&tree,int node,int st,int end,int l,int r,vector<int>&first){
    if(st>r||end<l){
        return -1;
    }
    if(l<=st&&end<=r){
        return tree[node];
    }
    int mid=(st+end)/2;
    int q1=query(euler,tree,2*node,st,mid,l,r,first);
    int q2=query(euler,tree,2*node+1,mid+1,end,l,r,first);
    if(q1!=-1&&q2!=-1){
        if(first[q1]<first[q2]){
            return q1;
        }
        else{
            return q2;
        }
    }
    else{
        return max(q1,q2);
    }
}
 
 
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	// freopen("iiiiiiiiiii.in", "r", stdin);
	// freopen("ooooooooooo.out", "w", stdout);
    int n,q;
    cin>>n>>q;
    vector<vector<int>>adj(n);
    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        u--;v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int>height(n,0);
    vector<int>first(n,0);
    vector<int>euler;
    dfs(adj,0,0,-1,euler,height,first);
    int cnt=euler.size();
    
    vector<int>tree(4*cnt,-1);
    build(euler,tree,1,0,cnt-1,first);
    
    while(q--){
        int u,v;
        cin>>u>>v;
        u--;
        v--;
        if(first[u]>first[v]){
            swap(u,v);
        }
        int lca=query(euler,tree,1,0,cnt-1,first[u],first[v],first);
        cout<<height[u]+height[v]-2*height[lca]<<endl;
        // cout<<u<<" "<<v<<" "<<lca<<endl;
    }

    return 0;
}




https://leetcode.com/problems/check-completeness-of-a-binary-tree/
bfs - while(q.front()!=NULL)   after loop ends if(q.size()>0) then not complete bt 
dfs - do number i has child 2*i and 2*i+1 use that for checking



https://www.interviewbit.com/problems/2sum-binary-tree/   get array out of it, or think inorder traversal in O(lohn)
https://www.interviewbit.com/problems/bst-iterator/
https://www.interviewbit.com/problems/inorder-traversal/
https://www.interviewbit.com/problems/preorder-traversal/
https://www.interviewbit.com/problems/postorder-traversal/
https://www.interviewbit.com/problems/max-edge-queries/
https://www.interviewbit.com/problems/least-common-ancestor/


https://www.geeksforgeeks.org/serialize-deserialize-binary-tree/
// preorder generally used , also -1 is used for null

https://codeforces.com/contest/1843/problem/F1