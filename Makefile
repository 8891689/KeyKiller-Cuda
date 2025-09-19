#---------------------------------------------------------------------
# Makefile
#---------------------------------------------------------------------

# 1. 原始碼與目標檔案定義 (Source and Object Definitions)
#---------------------------------------------------------------------
SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp Wildcard.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        GPU/GPUEngine.o Bech32.o Wildcard.o)

# 2. 編譯器與工具路徑 (Compilers and Tools)
#---------------------------------------------------------------------
CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc

# 3. 編譯與連結旗標 (Compilation and Linker Flags)
#---------------------------------------------------------------------

# --- 為多種 GPU 架構生成原生代碼 (Fat Binary) ---
# Pascal(10xx), Volta(V100), Turing(20xx), Ampere(30xx), Ada Lovelace(40xx)/Hopper
# 最後一行為 PTX JIT 編譯提供向前相容性
CUDA_ARCH = -gencode=arch=compute_61,code=sm_61 \
            -gencode=arch=compute_70,code=sm_70 \
            -gencode=arch=compute_75,code=sm_75 \
            -gencode=arch=compute_86,code=sm_86 \
            -gencode=arch=compute_90,code=sm_90 \
            -gencode=arch=compute_90,code=compute_90

# --- CPU (主機端) 編譯旗標 ---
ifdef debug
CXXFLAGS   = -static -msse4.1 -Wno-write-strings -g -I. -I$(CUDA)/include
else
# -O3: 最高級別優化
# -march=native: 為當前編譯機器的 CPU 架構生成最佳化指令
CXXFLAGS   = -static -msse4.1 -Wno-write-strings -O3 -march=native -I. -I$(CUDA)/include
endif

# --- GPU (設備端) NVCC 編譯旗標 ---
# Debug 版本: 啟用所有除錯資訊
NVCC_FLAGS_DEBUG = \
	-G \
	-g \
	--compiler-options -fPIC \
	-ccbin $(CXXCUDA) \
	-m64 \
	-I$(CUDA)/include \
	$(CUDA_ARCH)

# Release 版本: 啟用所有性能優化
NVCC_FLAGS_RELEASE = \
	-O3 \
	--use_fast_math \
	--fmad=true \
	-maxrregcount=0 \
	--ptxas-options=--allow-expensive-optimizations=true \
	--resource-usage \
	--compiler-options -fPIC \
	-ccbin $(CXXCUDA) \
	-m64 \
	-I$(CUDA)/include \
	$(CUDA_ARCH)
	# --- 可選的激進優化 (可能影響精度，請自行測試後啟用) ---
	--ftz=true
	--prec-div=false
	--prec-sqrt=false

# --- 連結旗標 ---
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart

# 4. 編譯規則 (Build Rules)
#---------------------------------------------------------------------

# --- 主要目標 ---
all: kk

kk: $(OBJET)
	@echo "==> Linking executable: kk..."
	$(CXX) $(OBJET) $(LFLAGS) -o kk
	@echo "==> Build finished: ./kk"

# --- GPU Kernel 編譯規則 ---
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
ifdef debug
	@echo "==> Compiling GPU Kernel (Debug): $<"
	$(NVCC) $(NVCC_FLAGS_DEBUG) --compile -o $@ -c $<
else
	@echo "==> Compiling GPU Kernel (Release): $<"
	$(NVCC) $(NVCC_FLAGS_RELEASE) --compile -o $@ -c $<
endif

# --- C++ 原始碼編譯規則 ---
$(OBJDIR)/%.o : %.cpp
	@echo "==> Compiling C++ Source: $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# 5. 目錄建立與清理 (Directory Creation & Clean Rules)
#---------------------------------------------------------------------
$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/GPU:
	@mkdir -p $(OBJDIR)/GPU

$(OBJDIR)/hash:
	@mkdir -p $(OBJDIR)/hash

clean:
	@echo "==> Cleaning project..."
	@rm -rf $(OBJDIR) kk
	@echo "==> Done."

.PHONY: all clean
