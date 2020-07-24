def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

from functools import reduce

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get_archs(n):
  archs = []
  for f in accel_asc(n):
    for factor in factors(f[0]):
      neurons = []
      i=0
      z0 = factor
      neurons.append(z0)
      while np.float(z0).is_integer():
        z0 = f[i]/z0
        neurons.append(np.int(z0))
        i+=1
        if i == len(f):
          archs.append(neurons)
          break
  return archs
