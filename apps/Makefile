# ======================================================================
# apps/Makefile
# ======================================================================

CURRENT_DIR ?= $(shell realpath .)
BUILD_DIR := $(CURRENT_DIR)/build
GEM5_ROOT := $(shell realpath $(APPS_ROOT)/..)

# 3. 工具链定义
CROSS_COMPILE ?= riscv64-unknown-elf-
CC      := $(CROSS_COMPILE)gcc
OBJDUMP := $(CROSS_COMPILE)objdump

MARCH ?= rv64gc
MABI  ?= lp64d
OPT_LEVEL ?= -O2

# 5. 编译选项
CFLAGS += -march=$(MARCH) -mabi=$(MABI)
CFLAGS += -static
CFLAGS += $(OPT_LEVEL)
CFLAGS += -g -Wall
CFLAGS += $(INCLUDES)

# ----------------------------------------------------------------------
# 4. gem5 仿真配置
# ----------------------------------------------------------------------
GEM5_BIN    := $(GEM5_ROOT)/build/RISCV/gem5.debug
SE_SCRIPT   := $(GEM5_ROOT)/configs/deprecated/example/se.py
GEM5_ARGS   := --cpu-type=O3CPU --caches --l2cache

# ======================================================================
# 目标定义
# ======================================================================
TARGET_BIN  := $(BUILD_DIR)/$(NAME)
TARGET_DUMP := $(TARGET_BIN).dump

# ======================================================================
# 构建规则
# ======================================================================

.PHONY: all clean help

all: $(TARGET_BIN) $(TARGET_DUMP)

# 创建构建目录
$(BUILD_DIR):
	@mkdir -p $@

# 编译规则
# 注意：这里的 $@ 和 $< 现在都是绝对路径
$(TARGET_BIN): $(SRCS) | $(BUILD_DIR)
	@echo "  CC      $@"
	$(CC) $(CFLAGS) $(SRCS) -o $@

# 反汇编规则
$(TARGET_DUMP): $(TARGET_BIN)
	@echo "  OBJDUMP $@"
	$(OBJDUMP) -d -S $< > $@

# 运行目标
run: $(TARGET_BIN)
# 	@echo "------------------------------------------------------------"
# 	@echo "  GEM5    Running Simulation..."
# 	@echo "  BIN:    $(TARGET_BIN)"
# 	@echo "------------------------------------------------------------"
	$(GEM5_BIN) $(SE_SCRIPT) $(GEM5_ARGS) --cmd=$(TARGET_BIN)

# 清理规则
clean:
	@echo "  CLEAN   $(BUILD_DIR)"
	rm -rf $(BUILD_DIR)

# 帮助信息 (用于调试路径是否正确)
help:
	@echo "Build system with Absolute Paths"
	@echo "Paths:"
	@echo "  CURRENT_DIR: $(CURRENT_DIR)"
	@echo "  BUILD_DIR:   $(BUILD_DIR)"
	@echo "Targets:"
	@echo "  Binary:      $(TARGET_BIN)"
	@echo "  Source:      $(SRCS)"