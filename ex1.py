import os
from matplotlib.pyplot import *
from numpy import *

#%matplotlib osx

out = "./out"

try:
    os.mkdir(out)
except:
    pass

phi = {
    "a": lambda x: exp(-x),
    "b": lambda x: (1+x)/(1+exp(x)),
    "c": lambda x: x + 1 - x * exp(x)
}

x = linspace(0,1)

for key in phi:
    close()
    xlabel(r"$x$")
    ylabel(r"$\Phi_1$")
    title(r"Serie 1%s)" % key)
    grid(True)
    plot(x, x)
    plot(x, phi[key](x))
    for ext in ["png"]: #, "pdf"]:
        savefig(os.path.join(out, "ex1_%s.%s" % (key, ext)))