LIBNAME    := plugin_swinQKVToContextPlugin
SOURCE_DIR := $(subst _,/,$(LIBNAME))
OUTPUT_DIR := $(subst _,/,$(LIBNAME))


# In order to properly compile the source files, included the enabled SM defines.
CUSTOM_CPP_FLAGS := $(ENABLED_SM_FLAGS)
CUSTOM_CU_FLAGS := $(ENABLED_SM_FLAGS)
include $(ROOT_PATH)/Makefile.component

