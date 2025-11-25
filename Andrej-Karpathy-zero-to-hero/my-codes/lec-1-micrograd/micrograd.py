import math
import numpy as np
import matplotlib.pyplot as plt
import random

# for drawing and visualizing the neural nets and mathematical expressions
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v.prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n.op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n.op, label = n.op)
      # and connect this node to it
      dot.edge(uid + n.op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2.op)

  return dot

class Value:
    def __init__(self , data , prev=() , op='' , label=''):
        self.data = data
        self.prev = set(prev)
        self.op = op
        self.label = label
        self.grad = 0
        self.backward = lambda : None


    def __repr__(self):
        return f"Value[ data = {self.data} , grad = {self.grad} ]"
    
    def __add__(self,other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data , (self , other),'+')
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward
        return out
    
   

    
    def __mul__(self,other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out.backward = backward
        return out
    
    def __radd__(self,other):
        return self+other
    
    def __rmul__(self,other):
        return self*other
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def backward():
            self.grad += (1 - t**2) * out.grad
        out.backward = backward
    
        return out
    

    def full_back(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v.prev:
                build_topo(child)
            topo.append(v)
        
        build_topo(self)

        for node in reversed(topo):
            node.grad = 0
        self.grad = 1

        for node in reversed(topo):
            node.backward()

    
        
# putting it together - condensed

class Neuron:
    def __init__(self , nin ):
        self.w = [Value(random.uniform(-1,1),label='w') for _ in range(nin)]
        self.b = Value(random.uniform(-1,1),label='b')

    def result(self , inputs):
        act = sum ((wi*xi for wi , xi in zip(self.w,inputs)) , self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin , nout):  
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def result(self , inputs):
        outs = [n.result(inputs) for n in self.neurons]
        return outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self,nin , nouts):
        self.num_neurons = [nin] + nouts
        self.layers = [Layer(self.num_neurons[i] , self.num_neurons[i+1]) for i in range(len(nouts)) ]

    def result(self,x):
        for layer in self.layers:
            x = layer.result(x)
        return x if len(x)!=1 else x[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

inputs = [2,4,5]
mlp = MLP(4,[2,5,4,1])
out = mlp.result(inputs)
out.full_back()
draw_dot(out)

# now to minimise loss using gradient descent

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets


mlp = MLP(3,[4,4,1])
ypred = [mlp.result(xi) for xi in xs]
loss = sum((ypredi + -1*ysi ) * (ypredi + -1*ysi )  for ypredi,ysi in zip(ypred,ys))
    


print("ys target" , ys)
print("y predictions" , ypred)
print("loss=" , loss)

for _ in range(80):
    ypred = [mlp.result(xi) for xi in xs]
    loss = sum((ypredi + -1*ysi ) * (ypredi + -1*ysi )  for ypredi,ysi in zip(ypred,ys))
    
    loss.full_back()

    for p in mlp.parameters():
        p.data += -0.008 * p.grad

    print(_+1 , loss  )
    

print("Final predictions:" , ypred)

# import random
# random.seed(42)

# # Assume Value, Neuron, Layer from your Python code (no MLP yet)
# inputs = [Value(1.0), Value(-2.0), Value(3.0)]

# layer = Layer(nin=3, nout=2)  # 3 inputs → 2 outputs
# outputs = layer.result(inputs)

# print("Python outputs:")
# for i, out in enumerate(outputs):
#     print(f"Neuron {i+1}: {out.data}")