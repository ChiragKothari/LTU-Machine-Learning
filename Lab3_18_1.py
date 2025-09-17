#  Exercise 18.1. For the following program, draw a UML class diagram that shows these classes and
#  the relationships among them.

# Parent Clas
class PingPongParent:
    pass

# Ping is a "PingPongParent", it has a "Pong" instance from the program below
class Ping(PingPongParent):
    def __init__(self, pong):
        self.pong = pong

# Pong is a "PingPongParent"
class Pong(PingPongParent):
    def __init__(self, pings=None):
        if pings is None:
            self.pings = []
        else:
            self.pings = pings
        def add_ping(self, ping):
            self.pings.append(ping)

pong = Pong()
ping = Ping(pong) # Ping has a Pong: instances of Ping contain references to instance of Pong
pong.add_ping(ping) # Pong has a Ping and a Pong: instances of Poing contain references to instance of Ping and instances of Pong