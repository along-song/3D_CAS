

class Solution:
    def commonChars(self , chars ):
        # write code here
        len1= len(chars)
        list1=list()
        for i in range(0,len1):
            list1.append(len(chars[i]))
        char1=chars[0]
        lenc1=len(char1)
        list2=list()
        for i in range(0,lenc1):
            T=True
            for j in range(0,len1):
                if(char1[i] not in chars[j]):
                    T=False
            if(T==True):
                list2.append(char1[i])
        S=""
        for item in list2:
            S=S+str(item)
        return S

A=Solution()
print(A.commonChars(["abdhs","whbdw","qysad"]))
