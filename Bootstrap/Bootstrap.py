
# coding: utf-8

# In[ ]:


class bootstraping():
#       """
#        initialization bootstrap parametrs
#        Bootstrap:bootstrap type
#        n: size of block
#        data: points cloud
#        m: bootstrap iteration
#     """
    def __call__(self,Bootstrap,n,data,m):
        self.Bootstrap=Bootstrap
        self.n=n
        self.data=data
        self.m=m
        
    def bootstraps(Bootstrap,n,data,m):
        bo=Bootstrap(n,data)
        result=[]
        for i in bo.bootstrap(m):
            result.append(i[0][0])
        return result
