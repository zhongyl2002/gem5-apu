GEM5_HOME := $(shell realpath $(CURRENT_DIR)/../../..)

CURRENT_DIR := $(shell realpath .)
APPS_ROOT := $(shell realpath $(CURRENT_DIR)/..)

NAME = hello
SRCS = $(CURRENT_DIR)/hello.c
MARCH := rv64gc
INCLUDES := -I$(CURRENT_DIR)
OPT_LEVEL := -O0

include $(APPS_ROOT)/Makefile