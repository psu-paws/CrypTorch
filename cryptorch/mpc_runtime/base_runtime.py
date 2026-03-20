from abc import ABC, abstractmethod

# List of functions that needs to be implemented in MPC
class BaseRuntime(ABC):
    @abstractmethod
    def init_runtime(self, rank, *args, **kwargs):
        pass
    @abstractmethod
    def get_comm_stats(self):
        pass
    @abstractmethod
    def encode(self, x, scale):
        pass
    @abstractmethod
    def encrypt(self, x, precision, src):
        pass
    @abstractmethod
    def decrypt(self, x, precision):
        pass
    @abstractmethod
    def decrypt_sequence(self, x, precisions, owners):
        pass
    @abstractmethod
    def ltz(self, x, *args, **kwargs):
        pass
    @abstractmethod
    def div(self, x, y):
        pass
    @abstractmethod
    def conv2d(self, x, y, stride, padding):
        pass
    @abstractmethod
    def mul(self, x, y):
        pass
    @abstractmethod
    def mul_(self, x, y):
        pass
    @abstractmethod
    def square(self, x):
        pass
    @abstractmethod
    def linear(self, x, y):
        pass
    @abstractmethod
    def square_(self, x):
        pass
    @abstractmethod
    def matmul(self, x, y):
        pass
    @abstractmethod
    def amax(self, x, dim, keepdim, *args, **kwargs):
        pass
    @abstractmethod
    def adaptive_avg_pool2d(self, x, output_size):
        pass
    @abstractmethod
    def max_pool2d(self, x, kernel_size, stride, padding, dilation, ceil_mode, *args, **kwargs):
        pass
