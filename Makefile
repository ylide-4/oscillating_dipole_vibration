# 使用的编译器
CC = icc

# 编译选项
CFLAGS = -qmkl -qopenmp -O3

# 源文件目录和目标文件目录
SRC_DIR = SRC
BIN_DIR = bin

# 源文件和目标可执行文件
TARGETS = edfft am_fft

# 定义规则
all: $(TARGETS)

# 编译规则
$(TARGETS): %: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $(BIN_DIR)/$@

# 清理规则
clean:
	rm -f $(BIN_DIR)/*
