import torch
import gemm_int
import time

print("test")
device = "cuda:0"
BS = 2
M = 2
N = 2
K = 2
A = torch.rand([BS, M, K]).to(device)
B = torch.rand([BS, K, N]).to(device)
A = (A * 2 ** 16).long()
B = (B * 2 ** 16).long()
#A = torch.ones([BS, M, K], dtype=torch.long).to(device) * 2 ** 40 - 1
#B = torch.ones([BS, K, N], dtype=torch.long).to(device) * 2 ** 40
A = A.type(torch.int64)
B = B.type(torch.int64)

print(A, B)
B = B
#C = torch.zeros([M, N]).to(device)
C = torch.zeros([BS, M, N], dtype=torch.int64).to(device)
C = C.type(torch.int64)

start_t = time.time()
gemm_int.matmul64(A, B.T, C, M, K, N, BS)
end_t = time.time()
#gemm64.cutlassGemm64(A, B.T, C, M, K, N)
print((end_t - start_t))
print(C)

C2 = torch.zeros([BS, M, N], dtype=torch.long).to(device)
for b in range(BS):
    for i in range(M):
        for j in range(N):
            C2[b][i][j] = (A[b, i, :] * B[b, :, j].T).sum()
'''
for i in range(M):
    for j in range(N):
        C2[i][j] = (A[i, :] * B[:, j].T).sum()
'''

print(C2)
print(((C - C2).abs() < 1e-4))
print("Passed ", len(((C - C2).abs() > 1e-4).nonzero()) == 0)

print("test done")
