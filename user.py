import pickle

class userr:
    def __init__(self, name, pas):
        self.n = name
        self.p = pas

def adduser(u,p):
    user=userr(u,p)
    pickle.dump(user, open("users.dat", "ab"))
    print(u+" added succesfully")

adduser('younes','younes')

objects = []
with (open("users.dat", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
           # print(objects)
        except EOFError:
            break

print(objects)