# Compiler
NVCC = nvcc

# Target executable name
TARGET = check_gpu

# Source file
SRC = check_gpu.cu

# Compilation flags
# NVCCFLAGS =

# Default rule
all: $(TARGET)

# Compile CUDA source file into an executable
$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)
	#$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)
