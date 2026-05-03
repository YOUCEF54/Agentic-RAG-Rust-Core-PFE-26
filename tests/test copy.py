import math
x,y = 0,0
i = 22
while i > 0:
    print(i)
    x+=(1/(365**2))*(i+1)
    i-=1
print((x)*365*100)

# for i in range(22):
#     x+=
