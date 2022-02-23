

str_in1=input(" ")
str_in2=input(" ")
A1=[int(n) for n in str_in1.split()]
A2=[int(n) for n in str_in2.split()]
len1=len(A1)
len2=len(A2)
list1=list()
list2=list()
list3=list()
for i in range(0,len1):
    list1.append(A1[i])
for i in range(0,len2):
    list2.append(A2[i])

for i in range(0,len1):
     if(A1[i] in list2):
         list3=list3+list(A1[i])
len3=len(list3)
for i in range(0,len3):
    list1.remove(list3[i])
list1.sort()

list4=list2+list1

print(list4)
