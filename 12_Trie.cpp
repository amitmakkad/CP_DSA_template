// Contents:-
// implement trie-insert, search, start_with
// implement trie-insert, search, isEmpty, remove
// max xor between two arrays 


//implement trie
class Node{
    public:
        Node *child[26];
        bool is_end;
        Node(){
            for(int i=0;i<26;i++){
                child[i]=NULL;
            }
            is_end=false;
        }
};
class Trie {
public:
    Node* root;
    Trie() {
        root=new Node();
    }
    
    void insert(string word) {
        Node* temp=root;
        for(char c:word){
            int idx=c-'a';
            if(temp->child[idx]==NULL){
                temp->child[idx]=new Node();
            }
            temp=temp->child[idx];
        }
        temp->is_end=true;
    }
    
    bool search(string word) {
        Node* temp=root;
        for(char c:word){
            int idx=c-'a';
            if(temp->child[idx]==NULL){
                return false;
            }
            temp=temp->child[idx];
        }
        return temp->is_end;
    }
    
    bool startsWith(string prefix) {
        Node* temp=root;
        for(char c:prefix){
            int idx=c-'a';
            if(temp->child[idx]==NULL){
                return false;
            }
            temp=temp->child[idx];
        }
        return true;
    }
};



 
#include <bits/stdc++.h> 
using namespace std; 

const int ALPHABET_SIZE = 26; 

 
struct TrieNode 
{ 
	struct TrieNode *children[ALPHABET_SIZE]; 
	bool isEndOfWord; 
}; 


struct TrieNode *getNode(void) 
{ 
	struct TrieNode *pNode = new TrieNode; 

	pNode->isEndOfWord = false; 

	for (int i = 0; i < ALPHABET_SIZE; i++) 
		pNode->children[i] = NULL; 

	return pNode; 
} 


void insert(struct TrieNode *root, string key) 
{ 
	struct TrieNode *pCrawl = root; 

	for (int i = 0; i < key.length(); i++) 
	{ 
		int index = key[i] - 'a'; 
		if (!pCrawl->children[index]) 
			pCrawl->children[index] = getNode(); 

		pCrawl = pCrawl->children[index]; 
	} 

	 
	pCrawl->isEndOfWord = true; 
} 


bool search(struct TrieNode *root, string key) 
{ 
	struct TrieNode *pCrawl = root; 

	for (int i = 0; i < key.length(); i++) 
	{ 
		int index = key[i] - 'a'; 
		if (!pCrawl->children[index]) 
			return false; 

		pCrawl = pCrawl->children[index]; 
	} 

	return (pCrawl != NULL && pCrawl->isEndOfWord); 
} 

bool isEmpty(TrieNode* root) 
{ 
    for (int i = 0; i < ALPHABET_SIZE; i++) 
        if (root->children[i]) 
            return false; 
    return true; 
    
} 

TrieNode* remove(TrieNode* root, string key, int i) 
{ 
    
    if (!root) 
        return NULL; 
  
    
    if (i == key.size()) { 
  
        
        if (root->isEndOfWord) 
            root->isEndOfWord = false; 
  
        
        if (isEmpty(root)) { 
            delete (root); 
            root = NULL; 
        } 
  
        return root;
    } 
  
    
    int index = key[i] - 'a'; 
    root->children[index] =  
          remove(root->children[index], key, i + 1); 
  
    
    if (isEmpty(root) && root->isEndOfWord == false) { 
        delete (root); 
        root = NULL; 
    } 
  
    return root; 
} 
 

 
int main()
{ 
	
	string keys[] = {"an", "and", "ant", "bad", "bat", "zoo"};  
	
	int n = sizeof(keys)/sizeof(keys[0]); 

	struct TrieNode *root = getNode(); 

	
	for (int i = 0; i < n; i++) 
		insert(root, keys[i]); 

	root = remove(root, "zoo", 0); 
	
	search(root, "zoo")? cout << "zoo --- " << "Yes\n" : 
						cout << "zoo --- " << "No\n"; 
	 
	return 0; 
} 




//Given two integer array A and B, you have to pick one element from each array such that their xor is maximum.
//  Return this maximum xor value.

struct TNN {
    TNN *left;
    TNN *right;
    TNN() : left(NULL), right(NULL) {}
};

void insert(TNN* head, int val){
    for(int i = 31;i>=0;i--){
        if(val&(1<<i)){
            // go to 1
            if(head->right == NULL) head->right = new TNN(); 
            head = head->right;
        }else{
            if(head->left == NULL) head->left = new TNN(); 
            head = head->left;
        }
    }
}
int tryout(int x, TNN* head){
    int ans = 0;
    for(int i = 31;i>=0;i--){
        if(x&(1<<i)){
            // go to zero
            if(head->left){
                ans += (1<<i);
                head = head->left;
            }else if(head->right){
                head = head->right;
            }else{
                return ans;
            }
        }else{
            // go to one
            if(head->right){
                ans += (1<<i);
                head = head->right;
            }else if(head->left){
                head = head->left;
            }else{
                return ans;
            }
        }
    }
    return ans;
}

int Solution::solve(vector<int> &A, vector<int> &B) {
    TNN* head = new TNN();
    for(auto x : A) insert(head, x);
    int best = 0;
    for(auto x : B) best = max(best, tryout(x, head));
    return best;
}
