LIBNAME    := plugin_swinQKVToContextPlugin_fmha_with_position_bias
SOURCE_DIR := plugin/swinQKVToContextPlugin/fmha_with_position_bias
OUTPUT_DIR := plugin/swinQKVToContextPlugin/fmha_with_position_bias

# Find out the SM's that are enabled and filter out the enabled SM's
# The SM's that are not enabled, exclude the SM specific files.
CHECK_SMS := 75 80 86 87 90
# if SM86 is specified, make sure SM80 is not filtered out as they are compatible.
ifeq ($(filter $(SMS),86),86)
    CHECK_SMS := $(filter-out 80,$(CHECK_SMS))
endif
$(eval EXCLUDE_FILES := $(call FILTER_SMS_FILES,$(SMS),$(CHECK_SMS),$(SOURCE_DIR)))
# In order to properly compile the source files, included the enabled SM defines.
CUSTOM_CPP_FLAGS := $(ENABLED_SM_FLAGS)
CUSTOM_CU_FLAGS := $(ENABLED_SM_FLAGS)
OLD_SMS:=$(SMS)
OLD_GENCODES := $(GENCODES)
SMS := $(filter $(AMPERE_FUSEDEPSEP_ARCH) $(LEGACY_FUSEDEPSEP_ARCH), $(SMS))
GENCODES :=
include $(MKINC_PATH)/gencodes.inc
plugin_swinQKVToContextPlugin_fmha_with_position_bias_GENCODES := $(GENCODES)

include $(ROOT_PATH)/Makefile.component
GENCODES := $(OLD_GENCODES)
SMS := $(OLD_SMS)
