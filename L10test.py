import pyopencl as cl
import numpy as np

KERNEL_SRC = """
__kernel void hello(__global int *out) {
    int i = get_global_id(0);
    out[i] = i * i;
}
"""

ctx   = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog  = cl.Program(ctx, KERNEL_SRC).build()

N = 8
out = np.zeros(N, dtype=np.int32)
out_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

prog.hello(queue, (N,), None, out_dev)
cl.enqueue_copy(queue, out, out_dev)
queue.finish()

print(out)     # → [ 0  1  4  9 16 25 36 49]