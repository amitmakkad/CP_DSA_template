// Contents:-
// partition ll into 2 ll based a value 
// ll sort 
// remove duplicates
// remove kth node from back
// reverse the order of all nodes at even positions
// find middle
// find reverse
// cycle detection
// copy a ll
// some q


// Given a linked list A and a value B, partition it such that all nodes less than B 
// come before nodes greater than or equal to B. 

ListNode* Solution::partition(ListNode* A, int B) {
    ListNode* small=new ListNode(0);
    ListNode* store_small=small;
    ListNode* large=new ListNode(0);
    ListNode* store_large=large;
    int x=B;
    
    while(A!=NULL){
        if(A->val<x){
            small->next=A;
            small=A;
        }
        else{
            large->next=A;
            large=A;
        }
        A=A->next;
    }
    
    large->next=NULL;
    
    small->next=store_large->next;
    return store_small->next;   
}

//cLL sort using merge sort
ListNode* merge(ListNode* A, ListNode* B) {
    if(B==NULL)return A;
    if(A==NULL)return B;

    if(A->val>B->val){
        ListNode* temp= A;
        A=B;
        B=temp;
    }
    ListNode* head=A;

    while(head!=NULL && head->next!=NULL && B!=NULL){
        if(head->next->val<B->val)
            head=head->next;
        else{
            ListNode* temps=head->next;
            head->next=B;
            B=B->next;
            head->next->next=temps;
            head=head->next;
        }
    }
    if(head->next==NULL)
        head->next=B;
    
    return A;
}
 
 
 ListNode* mergesort(ListNode* A){
    if (!A || !A->next) return A;
    
    ListNode *head, *a, *b, *slow, *fast, *ret;
    head = A;
    
    slow = head;
    fast = head;
    while (fast&&fast->next) {
        fast = fast->next->next;
        slow=slow->next;
    }
    a = head;
    b = slow->next;
    slow->next = NULL;
    
    a = mergesort(a);
    b = mergesort(b);
    
    if (!a) {
        return b;
    }
    if (!b) {
        return a;
    }
    return merge(a,b);
}
 
ListNode* Solution::sortList(ListNode* A) {
    
    return mergesort(A);
    
}


//Given a sorted linked list, delete all nodes that have duplicate numbers, 
// leaving only distinct numbers from the original list.For example,Given 1->2->3->3->4->4->5, return 1->2->5.

ListNode* Solution::deleteDuplicates(ListNode* A) {
    if(A==NULL){
        return A;
    }
    if(A->next==NULL){
        return A;
    }
    ListNode* head=NULL;
    ListNode* prev=NULL;
    
    while(A!=NULL){
        int cnt=0;
        while(A->next!=NULL){
            if(A->val==A->next->val){
                A=A->next;
                cnt++;
            }
            else{
                break;
            }
            
        }
        if(cnt==0){
            if(head==NULL){
                head=A;
            }
            prev=A;
            A=A->next;
        }
        else{
            if(prev!=NULL)
                prev->next=A->next;
            A=A->next;
        }   
    }
    return head;
    
}

//remove kth node from back
Make the first pointer go n nodes. Then move the second and first pointer simultaneously. 
This way, the first pointer is always ahead of the second pointer by n nodes. 

//Given a linked list A , reverse the order of all nodes at even positions.
ListNode* Solution::solve(ListNode* A) {
    if(A==NULL||A->next==NULL){
        return A;
    }
    ListNode *head=A;
    vector<int>v;
    while(A!=NULL&&A->next!=NULL){
        v.push_back(A->next->val);
        A=A->next->next;
    }    
    A=head;
    
    while(A!=NULL&&A->next!=NULL){
        A->next->val=v.back();
        v.pop_back();
        
        A=A->next->next;
    }
    return head;    
}

see here we used extra space
reverse a ll is O(1) space




//find middle efficient
void printMiddle(Node * head){
    if(head==NULL)return;
    Node *slow=head,*fast=head;
    while(fast!=NULL&&fast->next!=NULL){
        slow=slow->next;
        fast=fast->next->next;
    }
    cout<<slow->data;
}


// reverse a ll
node *reverse(node *head){
    if(head==NULL||head->next==NULL){
        return head;
    }
    else{
        node *prev=NULL;
        node *curr=head;
        while(curr!=NULL){
            prev=curr->prev;
            curr->prev=curr->next;
            curr->next=prev;
            curr=curr->prev;
        }
        return prev->prev;
    }
}

int main(){
    node *head=new node(10);
	node *temp1=new node(20);
	node *temp2=new node(30);
	head->next=temp1;
	temp1->prev=head;
	temp1->next=temp2;
	temp2->prev=temp1;
	printlist(head);
    head=reverse(head);
    printlist(head);
    return 0;
}







//cycle detection:-
// O(n^2) naive
// have viisited in struct 
// hashing
// all nodes pointing to dummy nodes, changes ll pointers
// floyd
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow=head;
        ListNode *fast=head;

        int cycle=0;
        while(slow!=NULL&&fast!=NULL&&fast->next!=NULL){
            slow=slow->next;
            fast=fast->next->next;
            if(slow==fast){
                cycle=1;
                break;
            }
        }
        if(cycle==0){
            return NULL;
        }
        slow=head;


        while(slow!=fast){
            slow=slow->next;
            fast=fast->next;
        }
        return slow;
    }
};



//copy a ll
Deep copy of ll with random pointer

Heres how the 1st algorithm goes.
Consider l1 as a node on the 1st list and l2 as the corresponding node on 2nd list.
Step 1:
Build the 2nd list by creating a new node for each node in 1st list. 
While doing so, insert each new node after its corresponding node in the 1st list.
Step 2:
The new head is the 2nd node as that was the first inserted node.
Step 3:
Fix the random pointers in the 2nd list: (Remember that l1->next is actually l2)
l2->random will be the node in 2nd list that corresponds l1->random, 
which is next node of l1->random.
Step 4:
Separate the combined list into 2: Splice out nodes that are part of second list. 
Return the new head that we saved in step 2.


RandomListNode* Solution::copyRandomList(RandomListNode *head) {
    if(head == NULL){
        return NULL;
    }

    // Step 1: create a new node for each existing node and join them together 
    // eg: A->B->C will be A->A'->B->B'->C->C'
    RandomListNode* node = head;
    while (node != NULL) {
        RandomListNode* copyNode = new RandomListNode(node->label);
        RandomListNode* nextNode = node->next;
        node->next = copyNode;
        copyNode->next = nextNode;
        node = nextNode;
    }

    // Step2: copy the random links: for each new node n', 
    // n'.random = n.random.next
    node = head;
    while(node != NULL) {
        RandomListNode* copyNode = node->next;
        if(node->random != NULL)
            copyNode->random = node->random->next;
        else
            copyNode->random = NULL;
        node = copyNode->next;
    }

    // Step3: detach the list: 
    // basically n.next = n.next.next; n'.next = n'.next.next
    node = head;
    RandomListNode* copyNode = node->next;
    RandomListNode* copyHead = head->next;
    while(copyNode != NULL && node != NULL) {
        node->next = node->next->next;
        if (copyNode->next == NULL) {
            break;
        }
        copyNode->next = copyNode->next->next;

        copyNode = copyNode->next;
        node = node->next;
    }

    return copyHead;
}


https://www.interviewbit.com/problems/reverse-alternate-k-nodes/
https://www.interviewbit.com/problems/k-reverse-linked-list/
https://www.interviewbit.com/problems/palindrome-list/
https://www.interviewbit.com/problems/insertion-sort-list/



