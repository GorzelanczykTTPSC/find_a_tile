import os
import subprocess


def getavg(n, args):
    suma = 0
    for i in range(n):
            sub = subprocess.run(args,  capture_output=True)
            time = sub.stdout.decode("utf-8")
            suma+=int(time)
    return suma/n


def threads_time(filename):
    f = open("threads-time.txt", "w")
    f.write(f"threads\ttime [ms]\n")
    for i in range(16):
        print(i+1)
        os.environ['OMP_NUM_THREADS'] = f"{i+1}"
        time = getavg(5, ["./bin/FindTilePar", filename, "50", "50", "400", "400"])
        f.write(f"{i+1}\t{time}\n")

    f.close()

def seq_par(filename, run):
    os.environ['OMP_NUM_THREADS'] = "16"
    f = open("seq_par.txt", "w")
    f.write(f"threads\tsequential[ms]\tparalel[ms]\n")
    for i in range(10, 110, 10):
        print(i)
        
        timeseq = getavg(2, ["./bin/FindTileSeq", filename, str(i), str(i), "400", "400", "1"])
        timepar = getavg(2, ["./bin/FindTilePar", filename, str(i), str(i), "400", "400", "1"])
        f.write(f"{i+1}\t{timeseq}\t{timepar}\n")

    f.close()


def cuda_omp(filename):
    os.environ['OMP_NUM_THREADS'] = "16"
    f = open("seq_par.txt", "w")
    f.write(f"threads\ttime[ms]\n")
    for i in range(10, 110, 10):
        print(i)
        
        timecuda = getavg(2, ["./cuda/build/bin/main", filename, str(i), str(i), "400", "400", "1"])
       
        f.write(f"{i+1}\t{timecuda}\n")

    f.close()

# threads_time("duck.jpg")
# seq_par("imgs/duck.jpg")
cuda_omp("imgs/duck.jpg")