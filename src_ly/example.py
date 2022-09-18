import time

time_start = time.time()

for _ in range(100000000):
    a = 10*100

time_end = time.time()

print(time_end-time_start)
